"""
mcp_tools/mcp_server.py
Agent 5 - MCP Server avec Auto-Correction Intégrée
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml

# Setup imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

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
    stages: Dict[str, Any]
    timestamp: str

# ============ AGENT 5 MCP SERVER ============

class Agent5MCPServer:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = parent_dir / "agent5_config.yaml"
        
        if not os.path.exists(config_path):
            print(f"⚠️ Config not found: {config_path}. Using defaults.")
            self.config = self._default_config()
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        print("🔧 Initializing Agent 5 MCP Server...")
        
        val_settings = self.config.get('validation', {})
        api_url = val_settings.get('mistral_api_url', 'http://localhost:1234/v1/chat/completions')
        self.max_retries = val_settings.get('max_retries', 3)

        self.validator = AdvancedHybridValidator(mistral_api_url=api_url)
        self.sandbox = SandboxExecutor()
        self.vm = VMExecutor(self.config.get('vm', {}))
        
        print("✅ Agent 5 MCP Server ready\n")

    def _default_config(self):
        return {
            'validation': {
                'mistral_api_url': 'http://localhost:1234/v1/chat/completions',
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
        
        return {
            "valid": validation.status == ValidationStatus.VALID,
            "status": str(validation.status.value),
            "score": int(validation.final_score),
            "errors": validation.semantic_errors,
            "warnings": validation.suggestions,
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
            "stages": {}
        }
        
        print(f"\n{'='*70}")
        print(f"MCP PIPELINE: {agent_name}")
        print(f"Command: {command}")
        print(f"Intent: {intent}")
        print(f"{'='*70}")
        
        # ========== STAGE 1: VALIDATION ==========
        print("\n[STAGE 1/4] VALIDATION")
        v_res = await self.validate_command(command, intent, agent_name)
        report['stages']['validation'] = v_res
        
        print(f"  Status: {v_res['status']}")
        print(f"  Score: {v_res['score']}/100")
        if v_res['errors']:
            print(f"  Errors: {v_res['errors']}")
        
        # ========== STAGE 2: AUTO-CORRECTION ==========
        correction_applied = False
        retry_count = 0
        
        while not v_res['valid'] and retry_count < self.max_retries:
            print(f"\n[STAGE 2/4] AUTO-CORRECTION (Attempt {retry_count + 1}/{self.max_retries})")
            
            corrected_cmd = self._auto_correct(command, v_res['errors'])
            
            if corrected_cmd == command:
                print("  ⚠️ No correction possible, stopping")
                break
            
            print(f"  Original: {command}")
            print(f"  Corrected: {corrected_cmd}")
            
            command = corrected_cmd
            correction_applied = True
            
            # Re-validate
            v_res = await self.validate_command(command, intent, f"{agent_name}-corrected-{retry_count+1}")
            print(f"  New Score: {v_res['score']}/100")
            
            retry_count += 1
        
        report['stages']['self_correction'] = {
            "applied": correction_applied,
            "attempts": retry_count,
            "final_command": command,
            "final_score": v_res['score']
        }
        
        # ========== STAGE 3: SANDBOX TEST ==========
        if v_res['valid']:
            if not skip_sandbox:
                print("\n[STAGE 3/4] SANDBOX TEST")
                
                sandbox_result = await self.sandbox.execute(
                    command=command.replace("TARGET", target),
                    timeout=self.config.get('docker', {}).get('timeout', 60)
                )
                
                report['stages']['sandbox'] = sandbox_result
                
                if sandbox_result['success']:
                    print(f"  ✅ Sandbox PASSED")
                else:
                    print(f"  ❌ Sandbox FAILED: {sandbox_result['errors']}")
                    report['final_status'] = 'failed_sandbox'
                    return report
            else:
                print("\n[STAGE 3/4] SANDBOX TEST - SKIPPED")
                report['stages']['sandbox'] = {"skipped": True}
        else:
            print("\n[STAGE 3/4] SANDBOX TEST - SKIPPED (validation failed)")
            report['stages']['sandbox'] = {"skipped": True, "reason": "validation_failed"}
            report['final_status'] = 'failed_validation'
            return report
        
        # ========== STAGE 4: VM EXECUTION ==========
        print("\n[STAGE 4/4] VM EXECUTION")
        
        try:
            with self.vm as vm:
                vm_result = vm.execute(
                    command=command.replace("TARGET", target),
                    target=target
                )
            
            report['stages']['vm_execution'] = vm_result
            
            if vm_result['success']:
                print(f"  ✅ VM Execution SUCCESS")
                report['final_status'] = 'success'
            else:
                print(f"  ❌ VM Execution FAILED")
                report['final_status'] = 'failed_vm'
        
        except Exception as e:
            print(f"  ❌ VM Connection Error: {e}")
            report['stages']['vm_execution'] = {"success": False, "errors": [str(e)]}
            report['final_status'] = 'vm_connection_error'
        
        print(f"\n{'='*70}")
        print(f"FINAL STATUS: {report['final_status']}")
        print(f"{'='*70}\n")
        
        return report

    def _auto_correct(self, cmd: str, errors: List[str]) -> str:
        """Auto-correction intelligente basée sur les erreurs détectées"""
        corrected = cmd.strip()
        
        # Détection: besoin de privilèges root
        needs_root = any(
            keyword in str(errors).lower() 
            for keyword in ["root", "privilege", "permission", "denied"]
        )
        
        if needs_root:
            # Option 1: Ajouter sudo si absent
            if not corrected.startswith("sudo"):
                print("    → Adding 'sudo' prefix")
                corrected = "sudo " + corrected
            
            # Option 2: Remplacer les scans nécessitant root
            else:
                # -sS → -sT (TCP connect scan, no root needed)
                if "-sS" in corrected:
                    print("    → Replacing -sS with -sT (no root needed)")
                    corrected = corrected.replace("-sS", "-sT")
                
                # Supprimer -O (OS detection needs root)
                if "-O" in corrected:
                    print("    → Removing -O (OS detection needs root)")
                    corrected = corrected.replace("-O", "")
                    corrected = " ".join(corrected.split())  # Clean double spaces
        
        # Correction: Target manquant
        if "TARGET" not in corrected and "target" in str(errors).lower():
            print("    → Adding TARGET placeholder")
            corrected += " TARGET"
        
        # Correction: Syntaxe nmap
        if corrected.startswith("nmap"):
            # Remove duplicate flags
            parts = corrected.split()
            seen = set()
            cleaned = []
            for part in parts:
                if part.startswith("-") and part in seen:
                    continue
                seen.add(part)
                cleaned.append(part)
            corrected = " ".join(cleaned)
        
        return corrected.strip()


# ============ FASTAPI APP ============

agent5 = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager (replaces deprecated on_event)"""
    global agent5
    try:
        print("🚀 Starting Agent 5 MCP Server...")
        agent5 = Agent5MCPServer()
        print("✅ MCP Server ready on http://0.0.0.0:5000")
    except Exception as e:
        print(f"❌ Critical error: {e}")
    yield
    print("👋 Shutting down MCP Server")

app = FastAPI(
    lifespan=lifespan,
    title="Agent 5 MCP Server",
    version="2.0",
    description="Validation, Auto-Correction, Sandbox & VM Execution"
)

@app.post("/mcp/validate", response_model=MCPValidateResponse)
async def mcp_validate(request: MCPValidateRequest):
    """Endpoint: Validation seule"""
    if not agent5:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    result = await agent5.validate_command(
        command=request.command,
        intent=request.intent,
        agent_name=request.agent_name,
        context=request.context
    )
    
    return MCPValidateResponse(**result)

@app.post("/mcp/execute", response_model=MCPExecuteResponse)
async def mcp_execute(request: MCPExecuteRequest):
    """Endpoint: Pipeline complet (Validation + Correction + Sandbox + VM)"""
    if not agent5:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    result = await agent5.execute_pipeline(
        command=request.command,
        intent=request.intent,
        target=request.target,
        agent_name=request.agent_name,
        skip_sandbox=request.skip_sandbox
    )
    
    return MCPExecuteResponse(
        final_status=result['final_status'],
        command=result['stages'].get('self_correction', {}).get('final_command', request.command),
        stages=result['stages'],
        timestamp=result['timestamp']
    )

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok" if agent5 else "initializing",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)