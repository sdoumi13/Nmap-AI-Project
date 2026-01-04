#!/usr/bin/env python3
"""
NMAP-AI ROUTER MAIN
Central entry point using corrected architecture

Architecture:
  User Query
    ↓
  RouterAgent (Agent 1)
    ├─ Comprehension Check
    ├─ Complexity Classification (Decide RAG vs Diffusion)
    ├─ Agent Selection (RAG or Diffusion)
    │  └─ Pure Command Generation (NO decision-making)
    ↓
  MCP Agent 5 (Central Authority)
    ├─ Validation (Hybrid Semantic + LLM)
    ├─ Auto-Correction (Loop with retries)
    ├─ Sandbox Test (Docker)
    ├─ VM Execution (SSH)
    └─ Structured Report
    ↓
  Response to User


Key Rules:
  ✅ RouterAgent = Decision-maker (Complexity)
  ✅ RAG/Diffusion = Pure generators (NO decisions)
  ✅ MCP Agent 5 = Executor (validation, correction, execution)
  ❌ NO agent bypasses MCP
  ❌ NO agent validates or executes directly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import asyncio
from agent_1_router.run_router import RouterAgent

# Configuration
COMPLEXITY_URL = "http://localhost:7000"
MCP_AGENT5_URL = "http://localhost:5000"

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
PURPLE = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


async def main():
    """Main entry point for RouterAgent"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{PURPLE}{BOLD}{'='*70}")
    print("      NMAP-AI CORRECTED ARCHITECTURE")
    print("      Query → Router (Complexity) → Agent → MCP (Validation) → Sandbox → VM")
    print(f"{'='*70}{RESET}\n")
    
    try:
        router = RouterAgent(
            complexity_url=COMPLEXITY_URL,
            mcp_url=MCP_AGENT5_URL
        )
    except Exception as e:
        print(f"{RED}Initialization error: {e}{RESET}")
        return
    
    print("-" * 70)
    print(f"Default Target: 192.168.188.128 (Ubuntu VM)")
    print(f"Complexity API: {COMPLEXITY_URL}/classify")
    print(f"MCP Agent 5: {MCP_AGENT5_URL}/mcp/execute")
    print("-" * 70)
    
    while True:
        try:
            user_input = input(f"\n{PURPLE}ROUTER > {RESET}")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input.strip():
                continue
            
            target = input(f"{YELLOW}Target (Default 192.168.188.128): {RESET}") or "192.168.188.128"
            
            print(f"\n{YELLOW}Processing...{RESET}")
            result = await router.route(user_input, target)
            
            # Display results
            if result["status"] == "rejected":
                print(f"\n{RED}╔══ REJECTED ══╗{RESET}")
                print(f"  {result['reason']}")
                print(f"{RED}╚═════════════╝{RESET}")
            
            elif result["status"] == "generation_failed":
                print(f"\n{RED}╔══ GENERATION FAILED ══╗{RESET}")
                print(f"  Agent: {result['agent']}")
                print(f"{RED}╚════════════════════════╝{RESET}")
            
            elif result["status"] == "executed":
                exec_res = result["execution"]
                
                # Generated command
                print(f"\n{GREEN}╔══ COMMAND ══╗{RESET}")
                print(f"  {result['command_generated']}")
                print(f"{GREEN}╚════════════════╝{RESET}")
                
                # Self-correction info
                if exec_res.get('stages', {}).get('self_correction', {}).get('applied'):
                    corr = exec_res['stages']['self_correction']
                    print(f"\n{YELLOW}╔══ CORRECTED ══╗{RESET}")
                    print(f"  {corr['final_command']}")
                    print(f"  Score: {corr.get('final_score', 'N/A')}/100")
                    print(f"  Attempts: {corr.get('attempts', 0)}")
                    print(f"{YELLOW}╚════════════════╝{RESET}")
                
                # Final status
                status = exec_res.get('final_status', 'unknown')
                color = GREEN if status == 'success' else RED
                
                print(f"\n{color}╔══ FINAL STATUS ══╗{RESET}")
                print(f"  Status: {status}")
                
                if status != 'success':
                    errors = exec_res.get('stages', {}).get('vm_execution', {}).get('errors', [])
                    if errors:
                        print(f"  Errors:")
                        for err in errors:
                            print(f"    - {err}")
                
                print(f"{color}╚═════════════════╝{RESET}")
                
                # Summary of execution stages
                stages = exec_res.get('stages', {})
                print(f"\n{CYAN}Execution Summary:{RESET}")
                if stages.get('validation'):
                    print(f"  ✓ Validation: {stages['validation'].get('status', 'N/A')}")
                if stages.get('self_correction'):
                    sc = stages['self_correction']
                    if sc.get('applied'):
                        print(f"  ✓ Auto-Correction: Applied ({sc.get('attempts', 0)} attempts)")
                    else:
                        print(f"  ✓ Auto-Correction: Not needed")
                if stages.get('sandbox'):
                    print(f"  ✓ Sandbox: {'PASSED' if stages['sandbox'].get('success') else 'SKIPPED/FAILED'}")
                if stages.get('vm_execution'):
                    print(f"  ✓ VM Execution: {'SUCCESS' if stages['vm_execution'].get('success') else 'FAILED'}")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()
    
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
