"""
Agent 5 - Complete Execution Script (FIXED)
Windows + GPU RTX 3080 + LM Studio + Docker + Ubuntu VM
"""

import asyncio
import os
import yaml
import json
from pathlib import Path
from datetime import datetime

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
        agent_name: str = "unknown"
    ) -> dict:
        
        print("="*70)
        print("AGENT 5 - VALIDATION & EXECUTION PIPELINE")
        print("="*70)
        print(f"Intent: {intent}")
        print(f"Command: {command}")
        print(f"Target: {target}")
        print(f"Agent: {agent_name}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*70)
        
        report = {
            "intent": intent,
            "original_command": command,
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
        
        # FIX: Convert ValidationStatus to string before serializing
        report['stages']['validation'] = {
            "status": str(validation['status']),  # Convert enum to string
            "score": validation['score'],
            "method": validation['method_used'],
            "errors": validation.get('errors', []),
            "warnings": validation.get('warnings', [])
        }
        
        # ============================================================
        # STAGE 2: SELF-CORRECTION (if needed)
        # ============================================================
        is_valid = validation.get('valid', False)
        is_recoverable = validation.get('status') == "recoverable"

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
                "history": history,
                "final_command": command,
                "final_validation_score": validation['score']
            }
            is_valid = validation.get('valid', False)
        else:
            print("\n[STAGE 2/4] SELF-CORRECTION")
            print("-"*70)
            print("  ✅ No correction needed")
            report['stages']['self_correction'] = {
                "applied": False,
                "reason": "validation passed" if is_valid else "invalid/unrecoverable"
            }
        
        # ============================================================
        # STAGE 3: SANDBOX TEST (Docker)
        # ============================================================
        if is_valid:
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
            
            if not sandbox_result['success']:
                report['final_status'] = 'failed_sandbox'
                return report
        else:
            print("\n[STAGE 3/4] SANDBOX TEST")
            print("-"*70)
            print("  ⭕ Skipped (validation failed)")
            report['stages']['sandbox'] = {"skipped": True}
            report['final_status'] = 'failed_validation'
            return report
        
        # ============================================================
        # STAGE 4: VM EXECUTION (Ubuntu SSH)
        # ============================================================
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
        
        # FINAL REPORT SUMMARY
        print("\n" + "="*70)
        print("FINAL REPORT SUMMARY")
        print("="*70)
        print(f"Status: {report['final_status']}")
        print(f"Final Command: {command}")
        print(f"Validation Score: {report['stages']['validation']['score']}/100")
        
        return report
    
    def _mock_correction(self, intent: str, failed_command: str, feedback: str) -> str:
        if "root" in str(feedback).lower() or "privileges" in str(feedback).lower():
            if "sudo" not in failed_command:
                return "sudo " + failed_command
        return "nmap -sT -p 80,443 TARGET"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution script - Agent 5 MCP Server (waits for requests from Router)"""
    
    # Initialize Agent 5
    print("\n✅ Agent 5 MCP Server ready and waiting for requests...")
    print("=" * 70)
    print("Expected flow:")
    print("  1. User enters query in ROUTER > prompt")
    print("  2. Router classifies complexity (RAG or Diffusion)")
    print("  3. Agent generates command")
    print("  4. Agent sends command to this Agent 5 MCP Server")
    print("  5. Agent 5 validates → corrects → sandbox → VM execution")
    print("=" * 70)
    
    agent5 = Agent5Pipeline(config_path="agent5_config.yaml")
    
    print("\n🎯 Agent 5 is running and ready to process commands from Router!")
    print("No static tests - commands come from Router based on user queries.")
    print("\nPress Ctrl+C to stop Agent 5 server...")
    
    # Keep the server running (in production, this would be the FastAPI server)
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Agent 5 server stopped.")


if __name__ == "__main__":
    try:
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   NMAP-AI AGENT 5                             ║
    ╚═══════════════════════════════════════════════════════════════╝
        """)
        
        if not os.path.exists("agent5_config.yaml"):
            raise FileNotFoundError("CRITICAL: 'agent5_config.yaml' is missing!")

        asyncio.run(main())
        
    except FileNotFoundError as fnf:
        print(f"\n❌ FILE ERROR: {fnf}")
        print("-> Please create agent5_config.yaml")
    except ImportError as imp:
        print(f"\n❌ IMPORT ERROR: {imp}")
        print("-> Check that your subfolders (validation, mcp, etc.) exist and contain __init__.py")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")