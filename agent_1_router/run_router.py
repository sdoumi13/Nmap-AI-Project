# Fichier: agent_1_router/run_router.py
"""
RouterAgent - Central Orchestrator
User Query → Complexity → Distributed RAG or Diffusion → MCP Agent 5 → Validation + Execution
"""

import sys
import os
import asyncio
import httpx
from pathlib import Path
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_1_router.comprehension import ComprehensionAgent
from agent_1_router.complexity import ComplexityAgent
from agent_1_router.distributed_routing import DistributedRAGClient

# Import Hybrid Validator from Agent 5
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_5_validation"))
try:
    from agent_5_validation.validation.hybrid_validator import AdvancedHybridValidator
    HYBRID_VALIDATOR_AVAILABLE = True
except ImportError:
    HYBRID_VALIDATOR_AVAILABLE = False

# Configuration
COLLEAGUE_RAG_URL = "http://192.168.1.218:8000"
MCP_AGENT5_URL = "http://localhost:5000"

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
PURPLE = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


class MCPClient:
    """Client for MCP Server (Agent 5)"""
    def __init__(self, mcp_url: str):
        self.base_url = mcp_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minutes
    
    async def execute_command(self, command: str, intent: str, target: str, 
                            agent_name: str) -> Dict[str, Any]:
        """Call MCP execute endpoint (validation + correction + sandbox + VM)"""
        try:
            print(f"  {CYAN}[MCP] Calling /mcp/execute...{RESET}")
            
            response = await self.client.post(
                f"{self.base_url}/mcp/execute",
                json={
                    "command": command,
                    "intent": intent,
                    "target": target,
                    "agent_name": agent_name,
                    "skip_sandbox": False
                }
            )
            response.raise_for_status()
            result = response.json()
            print(f"  {GREEN}[MCP] ✓ Response received{RESET}")
            return result
            
        except httpx.ConnectError as e:
            error_msg = f"Connection Error: {str(e)}"
            print(f"  {RED}[MCP] ✗ {error_msg}{RESET}")
            return {
                "final_status": "mcp_error",
                "command": command,
                "stages": {"error": error_msg},
                "timestamp": ""
            }
        except httpx.TimeoutException as e:
            error_msg = f"Timeout (exceeded {self.client.timeout}s): {str(e)}"
            print(f"  {RED}[MCP] ✗ {error_msg}{RESET}")
            return {
                "final_status": "mcp_error",
                "command": command,
                "stages": {"error": error_msg},
                "timestamp": ""
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            print(f"  {RED}[MCP] ✗ {error_msg}{RESET}")
            return {
                "final_status": "mcp_error",
                "command": command,
                "stages": {"error": error_msg},
                "timestamp": ""
            }
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"  {RED}[MCP] ✗ Unexpected error: {error_msg}{RESET}")
            return {
                "final_status": "mcp_error",
                "command": command,
                "stages": {"error": error_msg},
                "timestamp": ""
            }
    
    async def close(self):
        await self.client.aclose()

class RouterAgent:
    """
    Central Router:
    1. Checks comprehension
    2. Classifies complexity (Easy → RAG, Medium/Hard → Diffusion)
    3. Routes to appropriate agent
    4. Sends to MCP Agent 5 for execution
    """
    
    def __init__(self):
        print(f"{PURPLE}[*] Initializing RouterAgent...{RESET}")
        
        self.mcp_client = MCPClient(MCP_AGENT5_URL)
        self.comp_agent = ComprehensionAgent()
        self.complexity_agent = ComplexityAgent()
        
        # Initialize Hybrid Validator if available
        if HYBRID_VALIDATOR_AVAILABLE:
            try:
                self.validator = AdvancedHybridValidator(mistral_api_url="http://localhost:11434/v1/chat/completions")
                print(f"{GREEN}✓ Hybrid Validator initialized{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠ Hybrid Validator unavailable: {e}{RESET}")
                self.validator = None
        else:
            self.validator = None
        
        print(f"{GREEN}✓ RouterAgent ready{RESET}")
    
    async def route(self, user_query: str, target: str) -> Dict[str, Any]:
        """
        Main routing logic:
        1. Comprehension check
        2. Complexity classification
        3. Agent selection (RAG or Diffusion)
        4. Command generation
        5. MCP execution (validation + correction + sandbox + VM)
        """
        
        print(f"\n{CYAN}{'='*70}")
        print(f"ROUTER PIPELINE")
        print(f"{'='*70}{RESET}")
        
        # STEP 1: Comprehension Check
        print(f"\n{CYAN}[STEP 1/4] COMPREHENSION CHECK{RESET}")
        comp_result = self.comp_agent.analyze(user_query)
        
        if not comp_result['relevant']:
            print(f"  ❌ REJECTED. Score: {comp_result['score']:.2f}")
            print(f"     Reason: {comp_result['reason']}")
            return {
                "status": "rejected",
                "reason": comp_result['reason'],
                "score": comp_result['score']
            }
        
        print(f"  ✅ VALID. Score: {comp_result['score']:.2f}")
        
        # STEP 2: Complexity Classification
        print(f"\n{CYAN}[STEP 2/4] COMPLEXITY CLASSIFICATION{RESET}")
        complexity_result = self.complexity_agent.classify(user_query)
        
        level = complexity_result['level']
        level_color = GREEN if level == 'Easy' else YELLOW if level == 'Medium' else RED
        agent_choice = "RAG" if level == "Easy" else "DIFFUSION"
        
        print(f"  Level: {level_color}{level}{RESET}")
        print(f"  Confidence: {complexity_result['confidence']:.2f}")
        print(f"  Recommended Agent: {level_color}{agent_choice}{RESET}")
        print(f"  Reasoning: {complexity_result['reason']}")
        
        # STEP 3: Agent-Specific Command Generation
        print(f"\n{CYAN}[STEP 3/4] COMMAND GENERATION ({agent_choice}){RESET}")
        
        if agent_choice == "RAG":
            command = await self._generate_rag_command(user_query, target)
        else:
            command = await self._generate_diffusion_command(user_query, target)
        
        if not command:
            return {
                "status": "generation_failed",
                "agent": agent_choice
            }
        
        print(f"  Generated: {command}")
        
        # Pre-validate using Hybrid Validator (optional enhancement)
        if self.validator:
            print(f"\n  {CYAN}[Pre-Validation] Running hybrid validation...{RESET}")
            command = await self._validate_and_enhance_command(command, user_query, target)
        
        # STEP 4: MCP Execution (Agent 5 pipeline)
        print(f"\n{CYAN}[STEP 4/4] MCP EXECUTION (AGENT 5){RESET}")
        
        mcp_result = await self.mcp_client.execute_command(
            command=command,
            intent=user_query,
            target=target,
            agent_name=agent_choice.lower()
        )
        
        # Display the 4 STAGES from MCP result
        report = mcp_result.get('report', {})
        
        print(f"\n{YELLOW}{'─'*66}{RESET}")
        print(f"{YELLOW}MCP EXECUTION REPORT{RESET}")
        print(f"{YELLOW}{'─'*66}{RESET}")
        
        # STAGE 1: VALIDATION
        v_res = report.get('validation', {})
        if v_res:
            print(f"\n  {CYAN}[STAGE 1/4] VALIDATION{RESET}")
            status = v_res.get('status', 'unknown')
            status_color = GREEN if status == 'valid' else YELLOW if status == 'recoverable' else RED
            print(f"    Status: {status_color}{status.upper()}{RESET}")
            score = v_res.get('score', 0)
            print(f"    Score:  {score}/100")
            if v_res.get('errors'):
                print(f"    Issues: {', '.join(v_res['errors'][:2])}")
        
        # STAGE 2: AUTO-CORRECTION
        corr = report.get('self_correction', {})
        if corr:
            print(f"\n  {CYAN}[STAGE 2/4] AUTO-CORRECTION{RESET}")
            if corr.get('attempts'):
                print(f"    Original:  {corr.get('original_command', 'N/A')}")
                for attempt in corr.get('attempts', []):
                    print(f"    ├─ Attempt {attempt.get('iteration', '?')}: {attempt.get('fix', 'N/A')}")
                print(f"    Corrected: {corr.get('corrected_command', 'N/A')}")
                print(f"    Final Score: {corr.get('final_score', 'N/A')}/100")
            else:
                print(f"    Status: ✅ No correction needed")
                print(f"    Score:  {corr.get('final_score', 'N/A')}/100")
        
        # STAGE 3: SANDBOX EXECUTION
        sandbox = report.get('sandbox_execution', {})
        if sandbox:
            print(f"\n  {CYAN}[STAGE 3/4] SANDBOX EXECUTION{RESET}")
            print(f"    Command: {sandbox.get('command', command)}")
            exit_code = sandbox.get('exit_code', 'N/A')
            status_color = GREEN if exit_code == 0 else RED
            print(f"    Status: {status_color}{'✅ SUCCESS' if exit_code == 0 else '❌ FAILED'}{RESET}")
            print(f"    Exit Code: {exit_code}")
            print(f"    Runtime: {sandbox.get('runtime', 'N/A')}s")
            
            output_preview = sandbox.get('output', '')
            if output_preview:
                print(f"    Output Preview:")
                lines = output_preview.split('\n')[:4]
                for line in lines:
                    if line.strip():
                        print(f"      {line[:68]}")
        
        # STAGE 4: VM EXECUTION
        vm = report.get('vm_execution', {})
        if vm:
            print(f"\n  {CYAN}[STAGE 4/4] VM EXECUTION{RESET}")
            print(f"    Command: {vm.get('command', command)}")
            print(f"    Target: {vm.get('target', 'N/A')}")
            exit_code = vm.get('exit_code', 'N/A')
            status_color = GREEN if exit_code == 0 else RED
            print(f"    Status: {status_color}{'✅ SUCCESS' if exit_code == 0 else '❌ FAILED'}{RESET}")
            print(f"    Exit Code: {exit_code}")
            print(f"    Runtime: {vm.get('runtime', 'N/A')}s")
            
            output_preview = vm.get('output', '')
            if output_preview:
                print(f"    Output Preview:")
                lines = output_preview.split('\n')[:6]
                for line in lines:
                    if line.strip():
                        print(f"      {line[:68]}")
        
        # FINAL SUMMARY
        print(f"\n{YELLOW}{'─'*66}{RESET}")
        final_status = mcp_result.get('final_status', 'unknown')
        status_color = GREEN if final_status == 'success' else RED
        print(f"{status_color}{'✅ EXECUTION COMPLETED SUCCESSFULLY' if final_status == 'success' else '❌ EXECUTION FAILED'}{RESET}")
        print(f"{YELLOW}{'─'*66}{RESET}\n")
        
        return {
            "status": "executed",
            "complexity": complexity_result,
            "agent": agent_choice,
            "command_generated": command,
            "execution": mcp_result
        }
    
    async def _generate_rag_command(self, query: str, target: str) -> str:
        """
        Generate command using colleague's RAG Agent (Distributed).
        Sends query to colleague's machine (192.168.1.218:8000) via REST API.
        """
        try:
            # Import distributed client
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from agent_1_router.distributed_routing import DistributedRAGClient
            
            print(f"  {YELLOW}[Distributed Mode] Sending to colleague RAG (192.168.1.218:8000)...{RESET}")
            
            client = DistributedRAGClient(rag_url="http://192.168.1.218:8000")
            result = await client.generate_command(query=query, target=target)
            
            if result.get('status') == 'success':
                command = result.get('command')
                print(f"  {GREEN}[Colleague RAG] ✅ Command received: {command}{RESET}")
                
                # IMPORTANT: Ensure target is included in command
                command = self._ensure_target_in_command(command, target)
                
                # Also show validation info if available
                validation = result.get('validation', {})
                if validation.get('valid'):
                    print(f"    └─ Validated: Score {validation.get('score', 'N/A')}/100 ({validation.get('method', 'N/A')})")
                
                return command
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"  ❌ Colleague RAG Error: {error_msg}")
                return None
        except Exception as e:
            import traceback
            print(f"  ❌ Distributed RAG Exception: {e}")
            print(f"  {YELLOW}[Fallback] Using basic nmap command...{RESET}")
            # Fallback to basic command
            return f"nmap -sV {target}"
    
    async def _generate_diffusion_command(self, query: str, target: str) -> str:
        """
        Generate command using Diffusion Agent for MEDIUM/HARD queries.
        Diffusion is a pure generator model without decision logic.
        """
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "diffusion_models"))
            from discrete_diffusion_nmap import NmapDiscreteDiffusionLM, DiscreteDiffusionSampler
            
            print(f"  {YELLOW}[Diffusion] Generating command...{RESET}")
            
            model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
            sampler = DiscreteDiffusionSampler(model, max_steps=15)
            
            result = sampler.sample(query, verbose=False)
            command = result['final_command']
            
            # IMPORTANT: Enhance command based on intent (add missing flags)
            command = self._enhance_command_with_intent(command, query)
            
            # IMPORTANT: Ensure target is included
            command = self._ensure_target_in_command(command, target)
            
            print(f"  {GREEN}[Diffusion] ✅ Generated: {command}{RESET}")
            return command
            
        except Exception as e:
            print(f"  ❌ Diffusion Error: {e}")
            return None
    
    async def _validate_and_enhance_command(self, command: str, intent: str, target: str) -> str:
        """
        Validate command using Hybrid Validator and enhance if needed.
        Uses semantic + LLM validation to ensure command quality.
        """
        if not self.validator:
            # Validator not available, skip
            return command
        
        try:
            print(f"\n  {CYAN}[Pre-Validation] Checking command quality...{RESET}")
            
            # Run hybrid validation
            result = await self.validator.validate(
                command=command,
                intent=intent,
                agent_name="router"
            )
            
            score = result.final_score
            status = result.status
            
            print(f"    Validation Score: {score}/100")
            print(f"    Status: {status.value.upper()}")
            
            # If score is low, try enhancements
            if score < 80:
                enhanced = self._enhance_command_with_intent(command, intent)
                if enhanced != command:
                    command = enhanced
                    print(f"    Enhanced: {command}")
            
            return command
            
        except Exception as e:
            # If validation fails, continue with original command
            print(f"    ⚠️ Validation skipped: {str(e)[:50]}")
            return command
    
    def _ensure_target_in_command(self, command: str, target: str) -> str:
        """
        Ensure the command includes the target IP.
        Replaces placeholders (<target>, TARGET) and appends target if missing.
        """
        import re
        
        if not command:
            return command
        
        # First: Replace any placeholders
        command = command.replace('<target>', target)
        command = command.replace('<TARGET>', target)
        command = command.replace('TARGET', target)
        command = command.replace('target', target)
        
        # Second: Check if command already has a valid IP
        ip_pattern = r'\d+\.\d+\.\d+\.\d+'
        has_ip = re.search(ip_pattern, command)
        
        if has_ip:
            # Already has an IP address
            return command.strip()
        
        # Third: If it's a bare nmap command without target, append it
        if command.strip().startswith('nmap'):
            command = f"{command.strip()} {target}"
            print(f"    └─ Added target to command: {command}")
        
        return command.strip()
    
    def _enhance_command_with_intent(self, command: str, intent: str) -> str:
        """
        Enhance command based on intent keywords.
        Detects missing common flags and suggests adding them.
        """
        import re
        
        if not command or not intent:
            return command
        
        intent_lower = intent.lower()
        
        # Define intent keywords and their corresponding flags
        intent_flags = {
            'fragmentation': '-f',
            'fragment': '-f',
            'stealth': '-sS',
            'syn': '-sS',
            'fingerprint': '-O',
            'os detection': '-O',
            'version': '-sV',
            'service': '-sV',
            'script': '--script',
            'nse': '--script',
            'vulnerability': '--script vuln',
            'vuln': '--script vuln',
        }
        
        # Check which flags are missing
        for keyword, flag in intent_flags.items():
            if keyword in intent_lower:
                # Check if flag is already in command
                if flag not in command and flag.split()[0] not in command:
                    # Add the flag after 'nmap'
                    if command.strip().startswith('nmap'):
                        # Insert flag after 'nmap'
                        command = command.replace('nmap', f'nmap {flag}', 1)
                        print(f"    └─ Added flag '{flag}' based on intent: {command}")
        
        return command.strip()
    
    async def close(self):
        await self.mcp_client.close()


async def main():
    """Interactive shell for RouterAgent"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{PURPLE}{BOLD}{'='*70}")
    print("  NMAP-AI ROUTER")
    print("  Query → Comprehension → Complexity → Agent (RAG or Diffusion) → MCP Execution")
    print(f"{'='*70}{RESET}\n")
    
    try:
        router = RouterAgent()
    except Exception as e:
        print(f"{RED}Initialization Error: {e}{RESET}")
        return
    
    print(f"{YELLOW}─" * 70 + RESET)
    print(f"  Default Target: 192.168.188.128 (Ubuntu VM via SSH)")
    print(f"{YELLOW}─" * 70 + RESET)
    
    while True:
        try:
            user_input = input(f"\n{PURPLE}ROUTER > {RESET}")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print(f"{GREEN}👋 Goodbye!{RESET}")
                break
            
            if not user_input.strip():
                continue
            
            target_input = input(f"{YELLOW}Target [192.168.188.128]: {RESET}")
            target = target_input if target_input.strip() else "192.168.188.128"
            
            print(f"\n{YELLOW}Processing...{RESET}\n")
            result = await router.route(user_input, target)
            
            # Display results
            if result["status"] == "rejected":
                print(f"\n{RED}╔═ REJECTED ═╗{RESET}")
                print(f"  {result['reason']}")
                print(f"{RED}╚═════════════╝{RESET}")
                
            elif result["status"] == "generation_failed":
                print(f"\n{RED}╔═ GENERATION FAILED ═╗{RESET}")
                print(f"  Agent: {result['agent']}")
                print(f"{RED}╚═══════════════════════╝{RESET}")
                
            elif result["status"] == "executed":
                exec_res = result["execution"]
                
                print(f"\n{GREEN}╔═ COMMAND GENERATED ═╗{RESET}")
                print(f"  {result['command_generated']}")
                print(f"{GREEN}╚═════════════════════╝{RESET}")
                
                # Show correction if applied
                if exec_res.get('stages', {}).get('self_correction', {}).get('applied'):
                    corr = exec_res['stages']['self_correction']
                    print(f"\n{YELLOW}╔═ SELF-CORRECTION APPLIED ═╗{RESET}")
                    original = corr.get('original_command', result['command_generated'])
                    print(f"  Original: {original}")
                    print(f"  Corrected: {corr['final_command']}")
                    print(f"  Attempts: {corr.get('attempts', 0)}")
                    print(f"  Final Score: {corr.get('final_score', 'N/A')}/100")
                    print(f"{YELLOW}╚═══════════════════════════╝{RESET}")
                
                # Final result
                status = exec_res.get('final_status', 'unknown')
                color = GREEN if status == 'success' else RED
                
                print(f"\n{color}╔═ FINAL STATUS ═╗{RESET}")
                print(f"  {status.upper()}")
                
                # Handle MCP errors
                if status == 'mcp_error':
                    mcp_error = exec_res.get('stages', {}).get('error', 'Unknown MCP error')
                    print(f"  {RED}└─ MCP Error: {mcp_error}{RESET}")
                    print(f"\n  {YELLOW}Troubleshooting:{RESET}")
                    if 'Connection' in str(mcp_error):
                        print(f"    1. Check if Agent 5 is running:")
                        print(f"       python agent_5_validation/run_agent5.py")
                        print(f"    2. Verify port 5000 is accessible:")
                        print(f"       curl http://localhost:5000/health")
                    elif 'Timeout' in str(mcp_error):
                        print(f"    1. Agent 5 is slow or unresponsive")
                        print(f"    2. Check logs of agent_5_validation/run_agent5.py")
                        print(f"    3. Try again after checking VM connectivity")
                    elif 'HTTP' in str(mcp_error):
                        print(f"    1. MCP endpoint issue")
                        print(f"    2. Check agent_5_validation/mcp_tools/mcp_server.py")
                        print(f"    3. Verify POST /mcp/execute endpoint exists")
                
                # Handle VM execution errors
                elif status != 'success':
                    errors = exec_res.get('stages', {}).get('vm_execution', {}).get('errors', [])
                    if errors:
                        print(f"  {RED}└─ VM Execution Errors:{RESET}")
                        for err in errors[:3]:
                            print(f"    - {err}")
                    else:
                        other_error = exec_res.get('stages', {}).get('error', 'Unknown error')
                        if other_error:
                            print(f"  {RED}└─ Error: {other_error}{RESET}")
                
                print(f"{color}╚════════════════╝{RESET}")
        
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted{RESET}")
            break
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()
    
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())