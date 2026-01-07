"""
Agent 5 - Complete Execution Script with CORS FIXED
Windows + GPU RTX 3080 + LM Studio + Docker + Ubuntu VM
"""

import asyncio
import os
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# FastAPI imports
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import components
from validation.hybrid_validator import AdvancedHybridValidator, ValidationStatus
from mcp_tools.mcp_server import Agent5MCPServer
from execution.sandbox_executor import SandboxExecutor
from execution.vm_executor import VMExecutor
from self_correction.corrector import SelfCorrectionAgent


class Agent5Pipeline:
    """Pipeline complet Agent 5"""
    
    def __init__(self, config_path: str = "agent5_config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        print("⛄ SalamouAlaykom Initializing Agent 5 components...")
        
        # 1. Validator
        print("  [1/5] Advanced Hybrid Validator...")
        self.validator = AdvancedHybridValidator(
            mistral_api_url=self.config['validation']['mistral_api_url']
        )
        
        # 2. MCP Server (used locally as client)
        print("  [2/5] MCP Server/Client...")
        self.mcp_server = Agent5MCPServer()
        self.mcp_client = self.mcp_server
        
        # 3. Sandbox Executor
        print("  [3/5] Docker Sandbox...")
        self.sandbox = SandboxExecutor()
        
        # 4. VM Executor
        print("  [4/5] VM SSH Connection...")
        self.vm = VMExecutor(self.config['vm'])
        
        # 5. Self-Corrector
        print("  [5/5] Self-Correction Agent...")
        self.corrector = SelfCorrectionAgent(
            llm_generate_func=self._mock_correction,
            max_retries=self.config['validation']['max_retries']
        )
        
        print("✅ All components initialized!\n")
    
    async def process(
        self, 
        intent: str, 
        command: str, 
        target: str,
        agent_name: str = "unknown",
        skip_sandbox: bool = False
    ) -> dict:
        
        print("="*70)
        print("AGENT 5 - VALIDATION & EXECUTION PIPELINE")
        print("="*70)
        print(f"Intent: {intent}")
        print(f"Command: {command}")
        print(f"Target: {target}")
        print(f"Agent: {agent_name}")
        print(f"Skip Sandbox: {skip_sandbox}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*70)
        
        report = {
            "intent": intent,
            "original_command": command,
            "command": command,
            "target": target,
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # ============================================================
        # STAGE 1: VALIDATION via MCP
        # ============================================================
        print("\n[STAGE 1/4] VALIDATION VIA MCP")
        print("-"*70)
        
        # MCP Client returns a DICT
        validation = await self.mcp_client.validate_command(
            command=command,
            intent=intent,
            agent_name=agent_name
        )
        
        print(f"  Status: {validation['status']}") 
        print(f"  Score: {validation['score']}/100")
        print(f"  Method: {validation['method_used']}")
        
        if validation.get('errors'):
            print(f"  Errors: {validation['errors']}")
        
        # Convert ValidationStatus to string before serializing
        report['stages']['validation'] = {
            "status": str(validation['status']),
            "score": validation['score'],
            "method": validation['method_used'],
            "errors": validation.get('errors', []),
            "warnings": validation.get('warnings', [])
        }
        
        # ============================================================
        # STAGE 2: SELF-CORRECTION (if needed)
        # ============================================================
        is_valid = validation.get('valid', False)
        is_recoverable = str(validation.get('status')) == "recoverable"

        if not is_valid and is_recoverable:
            print("\n[STAGE 2/4] SELF-CORRECTION")
            print("-"*70)
            print("  Attempting to correct command...")
            
            corrected_cmd, history = await self.corrector.correct(
                intent=intent,
                failed_command=command,
                errors=validation.get('errors', []),
                mcp_client=self.mcp_client
            )
            
            print(f"  Correction history:")
            for entry in history:
                print(f"    - {entry}")
            
            command = corrected_cmd
            
            # Re-validate (returns dict)
            validation = await self.mcp_client.validate_command(
                command=command,
                intent=intent,
                agent_name=f"{agent_name}-corrected"
            )
            
            report['stages']['self_correction'] = {
                "applied": True,
                "original_command": report['original_command'],
                "corrected_command": command,
                "final_command": command,
                "history": history,
                "attempts": [{"iteration": i+1, "fix": h} for i, h in enumerate(history)],
                "final_score": validation['score']
            }
            report['command'] = command
            is_valid = validation.get('valid', False)
        else:
            print("\n[STAGE 2/4] SELF-CORRECTION")
            print("-"*70)
            print("  ✅ No correction needed")
            report['stages']['self_correction'] = {
                "applied": False,
                "final_score": validation['score'],
                "reason": "validation passed" if is_valid else "invalid/unrecoverable"
            }
        
        # ============================================================
        # STAGE 3: SANDBOX TEST (Docker)
        # ============================================================
        if is_valid and not skip_sandbox:
            print("\n[STAGE 3/4] SANDBOX TEST (Docker)")
            print("-"*70)
            print("  Executing in isolated Docker container...")
            
            sandbox_result = await self.sandbox.execute(
                command=command.replace("TARGET", target),
                timeout=self.config['docker'].get('timeout', 60)
            )
            
            if sandbox_result['success']:
                print(f"  ^_^ Sandbox test PASSED")
                print(f"  Execution time: {sandbox_result['time']:.2f}s")
                print(f"  Output preview: {sandbox_result['output'][:100]}...")
            else:
                print(f"  :| Sandbox test FAILED")
                print(f"  Errors: {sandbox_result['errors']}")
            
            report['stages']['sandbox'] = sandbox_result
            report['stages']['sandbox_execution'] = sandbox_result  # Alias for frontend
            
            if not sandbox_result['success']:
                report['final_status'] = 'failed_sandbox'
                return report
        else:
            print("\n[STAGE 3/4] SANDBOX TEST")
            print("-"*70)
            if skip_sandbox:
                print("  ⭕ Skipped (skip_sandbox=True)")
            else:
                print("  ⭕ Skipped (validation failed)")
            report['stages']['sandbox'] = {"skipped": True}
            report['stages']['sandbox_execution'] = {"skipped": True}
            
            if not is_valid:
                report['final_status'] = 'failed_validation'
                return report
        
        # ============================================================
        # STAGE 4: VM EXECUTION (Ubuntu SSH)
        # ============================================================
        if is_valid:
            print("\n[STAGE 4/4] VM EXECUTION (Ubuntu SSH)")
            print("-"*70)
            print(f"  Target: {target} | VM: {self.config['vm']['host']}")
            
            try:
                with self.vm as vm:
                    vm_result = vm.execute(command=command.replace("TARGET", target), target=target)
                
                if vm_result['success']:
                    print(f"  ^_^ VM execution SUCCESSFUL")
                else:
                    print(f"  :| VM execution FAILED")
                    print(f"  Errors: {vm_result['errors']}")
                
                report['stages']['vm_execution'] = vm_result
                report['final_status'] = 'success' if vm_result['success'] else 'failed_vm'
            
            except Exception as e:
                print(f" :| VM connection error: {e}")
                report['stages']['vm_execution'] = {"success": False, "errors": [str(e)]}
                report['final_status'] = 'vm_connection_error'
        else:
            report['stages']['vm_execution'] = {"skipped": True}
            report['final_status'] = 'failed_validation'
        
        # FINAL REPORT SUMMARY
        print("\n" + "="*70)
        print("FINAL REPORT SUMMARY")
        print("="*70)
        print(f"Status: {report['final_status']}")
        print(f"Final Command: {command}")
        print(f"Validation Score: {report['stages']['validation']['score']}/100")
        
        return report
    
    def _mock_correction(self, intent: str, failed_command: str, feedback: str) -> str:
        """Mock correction function for testing"""
        if "root" in str(feedback).lower() or "privileges" in str(feedback).lower():
            if "sudo" not in failed_command:
                return "sudo " + failed_command
        return "nmap -sT -p 80,443 TARGET"


# ============================================================================
# FASTAPI APPLICATION WITH CORS
# ============================================================================

app = FastAPI(title="Agent 5 MCP Server", version="1.0.0")

# CRITICAL: Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Vite dev server
        "http://localhost:5173",      # Alternative Vite port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize Agent 5 Pipeline
agent5 = None

@app.on_event("startup")
async def startup_event():
    global agent5
    print("\n" + "="*70)
    print("🚀 Starting Agent 5 MCP Server...")
    print("="*70)
    
    if not os.path.exists("agent5_config.yaml"):
        print("❌ ERROR: agent5_config.yaml not found!")
        return
    
    try:
        agent5 = Agent5Pipeline(config_path="agent5_config.yaml")
        print("✅ Agent 5 initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Agent 5: {e}")
        import traceback
        traceback.print_exc()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Agent 5 MCP Server",
        "status": "online",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "online",
        "local_agents": {
            "comprehension_ready": True,
            "rag_ready": True,
            "diffusion_ready": True
        },
        "external_services": {
            "agent5_mcp": {
                "status": "online",
                "url": "http://localhost:5000"
            }
        }
    }


@app.post("/mcp/validate")
async def mcp_validate(request: Request):
    """MCP Validate endpoint"""
    if agent5 is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 5 not initialized"}
        )
    
    try:
        data = await request.json()
        command = data.get("command")
        intent = data.get("intent")
        agent_name = data.get("agent_name", "unknown")
        
        validation_result = await agent5.mcp_client.validate_command(
            command=command,
            intent=intent,
            agent_name=agent_name
        )
        
        return validation_result
        
    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/mcp/execute")
async def mcp_execute(request: Request):
    """MCP Execute endpoint - Full pipeline"""
    if agent5 is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 5 not initialized"}
        )
    
    try:
        data = await request.json()
        command = data.get("command")
        intent = data.get("intent")
        target = data.get("target", "192.168.188.128")
        agent_name = data.get("agent_name", "unknown")
        skip_sandbox = data.get("skip_sandbox", False)
        
        print(f"\n📥 Received execution request:")
        print(f"   Intent: {intent}")
        print(f"   Command: {command}")
        print(f"   Target: {target}")
        
        result = await agent5.process(
            intent=intent,
            command=command,
            target=target,
            agent_name=agent_name,
            skip_sandbox=skip_sandbox
        )
        
        # Return with proper structure
        return {
            "final_status": result.get("final_status", "unknown"),
            "command": result.get("command"),
            "timestamp": result.get("timestamp"),
            "stages": result.get("stages", {}),
            "report": result.get("stages", {})  # Alias for frontend
        }
        
    except Exception as e:
        print(f"Execution error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "final_status": "error",
                "command": data.get("command", ""),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/history")
async def get_history():
    """Get execution history (mock for now)"""
    # TODO: Implement actual history storage
    return []


@app.get("/history/{entry_id}")
async def get_history_entry(entry_id: str):
    """Get specific history entry (mock for now)"""
    # TODO: Implement actual history retrieval
    return JSONResponse(
        status_code=404,
        content={"error": "History entry not found"}
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   NMAP-AI AGENT 5                             ║
    ║                   MCP SERVER WITH CORS                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if not os.path.exists("agent5_config.yaml"):
        print("❌ CRITICAL: 'agent5_config.yaml' is missing!")
        input("\nPress Enter to exit...")
        exit(1)
    
    try:
        # Run FastAPI server on port 5002 (to avoid conflicts)
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=5002,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Agent 5 server stopped by user.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")