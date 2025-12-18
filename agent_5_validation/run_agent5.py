"""
Agent 5 - Complete Execution Script
Windows + GPU RTX 3080 + LM Studio + Docker + Ubuntu VM
"""

import asyncio
import os
import yaml
from pathlib import Path
from datetime import datetime

# Import tous les composants
from validation.hybrid_validator import AdvancedHybridValidator, ValidationStatus
from mcp_tools.mcp_server import MCPServer
from mcp_tools.mcp_client import MCPClient
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
        print("⛄SalamouAlaykom Initializing Agent 5 components...")
        
        # 1. Validator
        print("  [1/5] Advanced Hybrid Validator...")
        self.validator = AdvancedHybridValidator(
            mistral_api_url="http://192.168.11.1:1234/v1/chat/completions"
        )
        
        # 2. MCP Server/Client
        print("  [2/5] MCP Server/Client...")
        self.mcp_server = MCPServer()
        self.mcp_client = MCPClient(self.mcp_server)
        
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
        
        print(" All components initialized!\n")
    
    async def process(
        self, 
        intent: str, 
        command: str, 
        target: str,
        agent_name: str = "unknown"
    ) -> dict:
        """
        Workflow complet:
        1. Validation MCP
        2. Self-correction si nécessaire
        3. Test Docker sandbox
        4. Exécution VM finale
        """
        
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
        
        validation = await self.mcp_client.validate_command(
            command=command,
            intent=intent,
            agent_name=agent_name
        )
        
        print(f"  Status: {validation.status}")
        print(f"  Score: {validation.score}/100")
        print(f"  Method: {validation.method_used}")
        
        if validation.errors:
            print(f"  Errors: {validation.errors}")
        
        report['stages']['validation'] = {
            "status": validation.status,
            "score": validation.score,
            "method": validation.method_used,
            "errors": validation.errors,
            "warnings": validation.warnings
        }
        
        # ============================================================
        # STAGE 2: SELF-CORRECTION (if needed)
        # ============================================================
        if not validation.valid and validation.status == "recoverable":
            print("\n[STAGE 2/4] SELF-CORRECTION")
            print("-"*70)
            print("  Attempting to correct command...")
            
            corrected_cmd, history = await self.corrector.correct(
                intent=intent,
                failed_command=command,
                errors=validation.errors,
                mcp_client=self.mcp_client
            )
            
            print(f"  Correction history:")
            for entry in history:
                print(f"    - {entry}")
            
            command = corrected_cmd
            
            # Re-validate
            validation = await self.mcp_client.validate_command(
                command=command,
                intent=intent,
                agent_name=f"{agent_name}-corrected"
            )
            
            report['stages']['self_correction'] = {
                "applied": True,
                "history": history,
                "final_command": command,
                "final_validation_score": validation.score
            }
        else:
            print("\n[STAGE 2/4] SELF-CORRECTION")
            print("-"*70)
            print("  ✅ No correction needed")
            
            report['stages']['self_correction'] = {
                "applied": False,
                "reason": "validation passed" if validation.valid else "invalid"
            }
        
        # ============================================================
        # STAGE 3: SANDBOX TEST (Docker)
        # ============================================================
        if validation.valid:
            print("\n[STAGE 3/4] SANDBOX TEST (Docker)")
            print("-"*70)
            print("  Executing in isolated Docker container...")
            
            sandbox_result = await self.sandbox.execute(
                command=command,
                timeout=self.config['docker'].get('timeout', 60)
            )
            
            if sandbox_result['success']:
                print(f"  ^_^ Sandbox test PASSED")
                print(f"  Execution time: {sandbox_result['time']:.2f}s")
                print(f"  Output preview:")
                print(f"    {sandbox_result['output'][:200]}...")
            else:
                print(f"  :| Sandbox test FAILED")
                print(f"  Errors: {sandbox_result['errors']}")
            
            report['stages']['sandbox'] = sandbox_result
            
            # Stop if sandbox failed
            if not sandbox_result['success']:
                report['final_status'] = 'failed_sandbox'
                return report
        
        else:
            print("\n[STAGE 3/4] SANDBOX TEST")
            print("-"*70)
            print("  ⏭  Skipped (validation failed)")
            
            report['stages']['sandbox'] = {"skipped": True}
            report['final_status'] = 'failed_validation'
            return report
        
        # ============================================================
        # STAGE 4: VM EXECUTION (Ubuntu SSH)
        # ============================================================
        print("\n[STAGE 4/4] VM EXECUTION (Ubuntu SSH)")
        print("-"*70)
        print(f"  Target: {target}")
        print(f"  VM: {self.config['vm']['host']}")
        print("  Executing via SSH...")
        
        try:
            with self.vm as vm:
                vm_result = vm.execute(command=command, target=target)
            
            if vm_result['success']:
                print(f"  ^_^ VM execution SUCCESSFUL")
                print(f"  Exit code: {vm_result['exit_code']}")
                print(f"  Output saved to report")
            else:
                print(f"  :| VM execution FAILED")
                print(f"  Exit code: {vm_result['exit_code']}")
                print(f"  Errors: {vm_result['errors']}")
            
            report['stages']['vm_execution'] = vm_result
            report['final_status'] = 'success' if vm_result['success'] else 'failed_vm'
        
        except Exception as e:
            print(f" :| VM connection error: {e}")
            report['stages']['vm_execution'] = {
                "success": False,
                "errors": [str(e)]
            }
            report['final_status'] = 'vm_connection_error'
        
        # ============================================================
        # FINAL REPORT
        # ============================================================
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        print(f"Status: {report['final_status']}")
        print(f"Final Command: {command}")
        print(f"Validation Score: {report['stages']['validation']['score']}/100")
        
        if report.get('stages', {}).get('vm_execution', {}).get('success'):
            print("\n :] Pipeline completed successfully!")
            print(f"\nNmap Output:\n{'-'*70}")
            print(report['stages']['vm_execution']['output'])
        else:
            print("\n :\\ Pipeline failed at some stage")
            print("Check report for details")
        
        return report
    
    def _mock_correction(self, intent: str, failed_command: str, feedback: str) -> str:
        # If the error is about root privileges, so i ADDed SUDO
        if "root" in str(feedback).lower() or "privileges" in str(feedback).lower():
            if "sudo" not in failed_command:
                return "sudo " + failed_command
        
        # Fallback for other errors (your existing logic)
        return "nmap -sT -p 80,443 TARGET"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution script"""
    
    # Initialize Agent 5
    agent5 = Agent5Pipeline(config_path="agent5_config.yaml")
    
    # Test cases
    test_cases = [
        {
            "name": "Test 1: Simple Web Scan",
            "intent": "Scan web ports",
            "command": "nmap -sT -p 80,443 TARGET",
            "target": "scanme.nmap.org",  # Safe test target
            "agent": "easy-rag"
        },
        {
            "name": "Test 2: Stealth Scan (needs correction)",
            "intent": "Stealth scan web ports",
            "command": "nmap -sS -p 80,443 TARGET",  # May need root
            "target": "192.168.188.1",  # Local network
            "agent": "medium-t5"
        },
        {
            "name": "Test 3: Full Scan",
            "intent": "Comprehensive security scan",
            "command": "nmap -sV -O -p- TARGET",
            "target": "192.168.188.1",
            "agent": "hard-diffusion"
        }
    ]
    
    # Run tests
    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'#'*70}")
        print(f"# {test['name']}")
        print(f"{'#'*70}\n")
        
        result = await agent5.process(
            intent=test['intent'],
            command=test['command'],
            target=test['target'],
            agent_name=test['agent']
        )
        
        # Save report
        import json
        report_file = f"agent5_report_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
        
        # Pause between tests
        if i < len(test_cases):
            print("\nWaiting 5 seconds before next test...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   NMAP-AI AGENT 5                             ║
    ╚═══════════════════════════════════════════════════════════════╝
        """)
        
        # Check if config exists before starting
        if not os.path.exists("agent5_config.yaml"):
            raise FileNotFoundError("CRITICAL: 'agent5_config.yaml' is missing!")

        asyncio.run(main())
        
    except FileNotFoundError as fnf:
        print(f"\n FILE ERROR: {fnf}")
        print("-> Please create agent5_config.yaml")
    except ImportError as imp:
        print(f"\n IMPORT ERROR: {imp}")
        print("-> Check that your subfolders (validation, mcp, etc.) exist and contain __init__.py")
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...") # Keeps window open to read errors