import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml

# Setup imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Note: Ensure these modules exist in your project structure
from validation.hybrid_validator import AdvancedHybridValidator, ValidationStatus
from execution.sandbox_executor import SandboxExecutor
from execution.vm_executor import VMExecutor

# ============ MODELS MCP ============

class MCPValidateRequest(BaseModel):
    command: str
    intent: str
    agent_name: str = "unknown"
    context: Optional[Dict[str, Any]] = None

class MCPValidateResponse(BaseModel):
    valid: bool
    status: str
    score: int
    errors: List[str] = []
    warnings: List[str] = []
    method_used: str
    timestamp: str

class MCPExecuteRequest(BaseModel):
    command: str
    intent: str
    target: str
    agent_name: str = "unknown"
    skip_sandbox: bool = False

class MCPExecuteResponse(BaseModel):
    final_status: str
    command: str
    intent: str
    original_command: str
    target: str
    agent: str
    timestamp: str
    stages: Dict[str, Any]

# ============ AGENT 5 MCP SERVER ============

class Agent5MCPServer:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = parent_dir / "agent5_config.yaml"
        
        self.config = self._load_config(config_path)
        
        print("🔧 Initializing Agent 5 MCP Server...")
        
        val_settings = self.config.get('validation', {})
        api_url = val_settings.get('mistral_api_url', 'http://192.168.11.1:1234/v1/chat/completions')
        self.max_retries = val_settings.get('max_retries', 3)
        
        self.validator = AdvancedHybridValidator(mistral_api_url=api_url)
        self.sandbox = SandboxExecutor()
        self.vm = VMExecutor(self.config.get('vm', {}))
        
        print("✅ Agent 5 MCP Server ready\n")

    def _load_config(self, path):
        if not os.path.exists(path):
            print(f"⚠️ Config not found: {path}. Using defaults.")
            return self._default_config()
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or self._default_config()
        except Exception as e:
            print(f"❌ Error loading YAML: {e}. Using defaults.")
            return self._default_config()

    def _default_config(self):
        return {
            'validation': {
                'mistral_api_url': 'http://192.168.11.1:1234/v1/chat/completions',
                'max_retries': 3
            },
            'vm': {
                'host': '192.168.188.128',
                'port': 22,
                'username': 'kali',
                'password': 'kali'
            },
            'docker': {
                'timeout': 60
            }
        }

    async def validate_command(
        self, 
        command: str, 
        intent: str, 
        agent_name: str, 
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Validation MCP endpoint"""
        validation = await self.validator.validate(
            command=command,
            intent=intent,
            agent_name=agent_name
        )
        
        # Handle both dict and object responses from validator
        if isinstance(validation, dict):
            status_val = validation.get('status')
            final_score = validation.get('score', validation.get('final_score', 0))
            semantic_errors = validation.get('errors', validation.get('semantic_errors', []))
            suggestions = validation.get('warnings', validation.get('suggestions', []))
        else:
            # If it's an object with attributes
            status_val = validation.status.value if hasattr(validation.status, 'value') else validation.status
            final_score = getattr(validation, 'final_score', getattr(validation, 'score', 0))
            semantic_errors = getattr(validation, 'semantic_errors', [])
            suggestions = getattr(validation, 'suggestions', [])
        
        return {
            "valid": status_val == "valid" or status_val == ValidationStatus.VALID,
            "status": str(status_val),
            "score": int(final_score),
            "errors": semantic_errors,
            "warnings": suggestions,
            "method_used": "hybrid_semantic_llm",
            "timestamp": datetime.now().isoformat()
        }

    async def execute_pipeline(
        self, 
        command: str, 
        intent: str, 
        target: str, 
        agent_name: str, 
        skip_sandbox: bool = False
    ) -> Dict[str, Any]:
        """Pipeline complet: Validation → Auto-Correction → Sandbox → VM"""
        
        report = {
            "command": command,
            "intent": intent,
            "target": target,
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "final_status": "unknown",
            "stages": {}
        }
        
        print(f"\n{'='*70}\nMCP PIPELINE: {agent_name}\nCommand: {command}\n{'='*70}")
        
        # ========== STAGE 1: VALIDATION ==========
        v_res = await self.validate_command(command, intent, agent_name)
        report['stages']['validation'] = v_res
        
        # ========== STAGE 2: AUTO-CORRECTION ==========
        correction_applied = False
        retry_count = 0
        
        while not v_res['valid'] and retry_count < self.max_retries:
            print(f"\n[STAGE 2] AUTO-CORRECTION (Attempt {retry_count + 1}/{self.max_retries})")
            corrected_cmd = self._auto_correct(command, v_res['errors'])
            
            if corrected_cmd == command:
                break
            
            command = corrected_cmd
            correction_applied = True
            v_res = await self.validate_command(command, intent, f"{agent_name}-retry-{retry_count+1}")
            retry_count += 1
        
        report['stages']['self_correction'] = {
            "applied": correction_applied,
            "final_command": command,
            "attempts": retry_count
        }

        if not v_res['valid']:
            report['final_status'] = 'failed_validation'
            return report
        
        # ========== STAGE 3: SANDBOX TEST ==========
        if not skip_sandbox:
            sandbox_result = await self.sandbox.execute(
                command=command.replace("TARGET", target),
                timeout=self.config.get('docker', {}).get('timeout', 60)
            )
            report['stages']['sandbox'] = sandbox_result
            if not sandbox_result.get('success'):
                report['final_status'] = 'failed_sandbox'
                return report
        
        # ========== STAGE 4: VM EXECUTION ==========
        try:
            # Using context manager for VM connection
            with self.vm as vm:
                vm_result = vm.execute(
                    command=command.replace("TARGET", target),
                    target=target
                )
            report['stages']['vm_execution'] = vm_result
            report['final_status'] = 'success' if vm_result.get('success') else 'failed_vm'
        except Exception as e:
            report['final_status'] = 'vm_connection_error'
            report['stages']['vm_execution'] = {"success": False, "errors": [str(e)]}
        
        return report

    def _auto_correct(self, cmd: str, errors: List[str]) -> str:
        corrected = cmd.strip()
        err_str = str(errors).lower()
        
        # Logic: Root privileges
        if any(k in err_str for k in ["root", "privilege", "permission"]):
            if not corrected.startswith("sudo"):
                corrected = "sudo " + corrected
            elif "-sS" in corrected:
                corrected = corrected.replace("-sS", "-sT")
        
        # Logic: Placeholder missing
        if "target" in err_str and "TARGET" not in corrected:
            corrected += " TARGET"
            
        return corrected

# ============ FASTAPI APP ============

agent5_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent5_instance
    print("🚀 Starting Agent 5 MCP Server...")
    agent5_instance = Agent5MCPServer()
    yield
    print("👋 Shutting down MCP Server")

app = FastAPI(lifespan=lifespan, title="Agent 5 MCP Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/mcp/validate", response_model=MCPValidateResponse)
async def mcp_validate(request: MCPValidateRequest):
    if not agent5_instance:
        raise HTTPException(status_code=503, detail="Agent not ready")
    result = await agent5_instance.validate_command(request.command, request.intent, request.agent_name)
    return MCPValidateResponse(**result)

@app.post("/mcp/execute", response_model=MCPExecuteResponse)
async def mcp_execute(request: MCPExecuteRequest):
    if not agent5_instance:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    result = await agent5_instance.execute_pipeline(
        command=request.command,
        intent=request.intent,
        target=request.target,
        agent_name=request.agent_name,
        skip_sandbox=request.skip_sandbox
    )
    
    # Map report to the Pydantic response model
    return MCPExecuteResponse(
        final_status=result['final_status'],
        command=result['stages'].get('self_correction', {}).get('final_command', request.command),
        intent=request.intent,
        original_command=request.command,
        target=request.target,
        agent=request.agent_name,
        timestamp=result['timestamp'],
        stages=result['stages']
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)