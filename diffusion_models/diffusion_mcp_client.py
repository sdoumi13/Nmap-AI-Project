"""
agent3_diffusion_mcp_client.py
Diffusion Agent avec MCP Client intégré
"""

import os
import time
import asyncio
import httpx
from typing import Dict, Any
from discrete_diffusion_nmap import NmapDiscreteDiffusionLM, DiscreteDiffusionSampler

# URLs
COMPLEXITY_API_URL = "http://localhost:7000"
MCP_AGENT5_URL = "http://localhost:5000"

# Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
PURPLE = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ComplexityClient:
    """Client pour classification de complexité (Agent 1)"""
    
    def __init__(self, api_url: str):
        self.base_url = api_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def classify(self, query: str) -> Dict[str, Any]:
        try:
            response = await self.client.post(
                f"{self.base_url}/classify",
                json={"query": query, "user_id": "diffusion-agent"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {
                "complexity": "HARD",
                "confidence": 1.0,
                "recommended_agent": "DIFFUSION",
                "reasoning": f"API error, defaulting to DIFFUSION: {str(e)}"
            }
    
    async def close(self):
        await self.client.aclose()


class MCPClient:
    """MCP Client pour Agent 5 (Validation + Exécution)"""
    
    def __init__(self, agent5_url: str):
        self.base_url = agent5_url
        self.client = httpx.AsyncClient(timeout=180.0)  # Long timeout pour VM
    
    async def validate_command(
        self, 
        command: str, 
        intent: str, 
        agent_name: str = "diffusion-agent"
    ) -> Dict[str, Any]:
        """Validation seule via MCP"""
        try:
            response = await self.client.post(
                f"{self.base_url}/mcp/validate",
                json={
                    "command": command,
                    "intent": intent,
                    "agent_name": agent_name,
                    "context": {"source": "diffusion"}
                }
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPError as e:
            return {
                "valid": False,
                "status": "mcp_error",
                "score": 0,
                "errors": [f"MCP connection error: {str(e)}"],
                "warnings": [],
                "method_used": "none"
            }
    
    async def execute_command(
        self,
        command: str,
        intent: str,
        target: str,
        agent_name: str = "diffusion-agent",
        skip_sandbox: bool = False
    ) -> Dict[str, Any]:
        """Pipeline complet via MCP (Validation + Correction + Sandbox + VM)"""
        try:
            response = await self.client.post(
                f"{self.base_url}/mcp/execute",
                json={
                    "command": command,
                    "intent": intent,
                    "target": target,
                    "agent_name": agent_name,
                    "skip_sandbox": skip_sandbox
                }
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPError as e:
            return {
                "final_status": "mcp_error",
                "command": command,
                "stages": {"error": str(e)},
                "timestamp": ""
            }
    
    async def close(self):
        await self.client.aclose()


class DiffusionAgentWithMCP:
    """Diffusion Agent avec Complexity Routing + MCP Integration"""
    
    def __init__(
        self, 
        model_checkpoint: str,
        complexity_url: str,
        mcp_url: str
    ):
        print(f"{PURPLE}[*] Loading Diffusion Model...{RESET}")
        
        # Load model
        if not os.path.exists(model_checkpoint):
            print(f"{YELLOW}⚠ Checkpoint not found, using base model{RESET}")
            self.model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
        else:
            self.model = NmapDiscreteDiffusionLM(model_name=model_checkpoint, use_adapter=False)
        
        self.sampler = DiscreteDiffusionSampler(self.model, max_steps=15)
        
        # Clients
        self.complexity_client = ComplexityClient(complexity_url)
        self.mcp_client = MCPClient(mcp_url)
        
        print(f"{GREEN}✓ Diffusion Model + MCP Client ready{RESET}")
    
    async def process_query(
        self, 
        user_query: str,
        target: str,
        mode: str = "validate",  # "validate" or "execute"
        force_diffusion: bool = False
    ) -> Dict[str, Any]:
        """
        Pipeline complet:
        1. Complexity Classification
        2. Diffusion Generation
        3. MCP Validation/Execution
        """
        
        # ========== STEP 1: COMPLEXITY ==========
        print(f"{YELLOW}[1/3] Analyzing query complexity...{RESET}")
        complexity_result = await self.complexity_client.classify(user_query)
        
        print(f"\n{PURPLE}╔══ COMPLEXITY ANALYSIS ══╗{RESET}")
        print(f"  Level: {complexity_result['complexity']}")
        print(f"  Confidence: {complexity_result['confidence']:.2f}")
        print(f"  Recommended: {complexity_result['recommended_agent']}")
        print(f"  Reason: {complexity_result['reasoning']}")
        print(f"{PURPLE}╚═════════════════════════╝{RESET}\n")
        
        # Check routing
        if not force_diffusion and complexity_result['recommended_agent'] != 'DIFFUSION':
            return {
                "status": "routed_to_rag",
                "complexity": complexity_result,
                "message": f"Query routed to RAG due to {complexity_result['complexity']} complexity"
            }
        
        # ========== STEP 2: DIFFUSION GENERATION ==========
        print(f"{YELLOW}[2/3] Generating via Diffusion...{RESET}")
        
        gen_start = time.time()
        diffusion_result = self.sampler.sample(user_query, verbose=False)
        gen_time = time.time() - gen_start
        
        command = diffusion_result['final_command']
        
        print(f"{GREEN}✓ Generated in {gen_time:.2f}s ({diffusion_result['steps']} steps){RESET}")
        print(f"{PURPLE}  Trajectory: {' → '.join(diffusion_result['trajectory'][:3])} ... → {command}{RESET}\n")
        
        # ========== STEP 3: MCP ==========
        print(f"{YELLOW}[3/3] Processing via MCP (Agent 5)...{RESET}")
        
        if mode == "validate":
            # Validation seule
            mcp_result = await self.mcp_client.validate_command(
                command=command,
                intent=user_query,
                agent_name="diffusion-agent"
            )
            
            return {
                "status": "validated",
                "complexity": complexity_result,
                "command": command,
                "diffusion": {
                    "steps": diffusion_result['steps'],
                    "trajectory": diffusion_result['trajectory'],
                    "time": gen_time
                },
                "validation": mcp_result
            }
        
        else:  # mode == "execute"
            # Pipeline complet (avec auto-correction, sandbox, VM)
            mcp_result = await self.mcp_client.execute_command(
                command=command,
                intent=user_query,
                target=target,
                agent_name="diffusion-agent"
            )
            
            return {
                "status": "executed",
                "complexity": complexity_result,
                "command": command,
                "diffusion": {
                    "steps": diffusion_result['steps'],
                    "trajectory": diffusion_result['trajectory'],
                    "time": gen_time
                },
                "execution": mcp_result
            }
    
    async def close(self):
        await self.complexity_client.close()
        await self.mcp_client.close()


async def interactive_shell():
    """Shell interactif"""
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{PURPLE}{BOLD}" + "="*70)
    print("      DIFFUSION AGENT v3.0 - MCP CLIENT")
    print("      Query → Complexity → Diffusion → MCP → Agent 5")
    print("="*70 + f"{RESET}\n")
    
    print(f"{YELLOW}[*] Initializing...{RESET}")
    
    try:
        agent = DiffusionAgentWithMCP(
            model_checkpoint="nmap_diffusion_checkpoint",
            complexity_url=COMPLEXITY_API_URL,
            mcp_url=MCP_AGENT5_URL
        )
        print(f"{GREEN}[✓] Diffusion Agent + MCP ready!{RESET}\n")
    except Exception as e:
        print(f"{RED}[✗] Init error: {e}{RESET}")
        return
    
    print("-" * 70)
    print("Commands:")
    print("  • Natural language query")
    print("  • 'validate' : Validation mode")
    print("  • 'execute' : Full execution mode (sandbox + VM)")
    print("  • 'force' : Toggle force-diffusion")
    print("  • 'exit' : Quit")
    print("-" * 70)
    
    mode = "validate"
    force_diffusion = False
    
    while True:
        try:
            user_input = input(f"\n{PURPLE}{BOLD}DIFFUSION [{mode}{'|FORCED' if force_diffusion else ''}] > {RESET}")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if user_input.lower() == 'validate':
                mode = "validate"
                print(f"{GREEN}Mode: Validation only{RESET}")
                continue
            
            if user_input.lower() == 'execute':
                mode = "execute"
                print(f"{YELLOW}Mode: Full execution{RESET}")
                continue
            
            if user_input.lower() == 'force':
                force_diffusion = not force_diffusion
                print(f"{YELLOW}Force Diffusion: {'ON' if force_diffusion else 'OFF'}{RESET}")
                continue
            
            if not user_input.strip():
                continue
            
            # Get target
            target = input(f"{YELLOW}Target (default: 192.168.1.0/24): {RESET}") or "192.168.1.0/24"
            
            # Process
            print(f"\n{YELLOW}[...] Processing...{RESET}\n")
            start = time.time()
            
            result = await agent.process_query(
                user_query=user_input,
                target=target,
                mode=mode,
                force_diffusion=force_diffusion
            )
            
            duration = time.time() - start
            
            # Display results
            if result["status"] == "routed_to_rag":
                print(f"\n{BLUE}╔══ ROUTED TO RAG ══╗{RESET}")
                print(f"  {result['message']}")
                print(f"{BLUE}╚════════════════════╝{RESET}")
                continue
            
            # Command
            print(f"\n{GREEN}╔══ DIFFUSION GENERATED ══╗{RESET}")
            print(f"{BOLD}  {result['command']}{RESET}")
            print(f"  Steps: {result['diffusion']['steps']}")
            print(f"  Time: {result['diffusion']['time']:.2f}s")
            print(f"{GREEN}╚═════════════════════════╝{RESET}")
            
            # Validation
            if "validation" in result:
                val = result["validation"]
                color = GREEN if val["valid"] else RED
                
                print(f"\n{color}╔══ MCP VALIDATION ══╗{RESET}")
                print(f"  Status: {val['status']}")
                print(f"  Score: {val['score']}/100")
                
                if val.get("errors"):
                    print(f"{RED}  Errors:{RESET}")
                    for err in val["errors"]:
                        print(f"    • {err}")
                
                print(f"{color}╚════════════════════╝{RESET}")
            
            # Execution
            if "execution" in result:
                exec_res = result["execution"]
                
                # Correction info
                if exec_res['stages'].get('self_correction', {}).get('applied'):
                    corr = exec_res['stages']['self_correction']
                    print(f"\n{YELLOW}╔══ AUTO-CORRECTION ══╗{RESET}")
                    print(f"  Attempts: {corr['attempts']}")
                    print(f"  Final: {corr['final_command']}")
                    print(f"  Score: {corr['final_score']}/100")
                    print(f"{YELLOW}╚═════════════════════╝{RESET}")
                
                # Final status
                status_color = GREEN if exec_res['final_status'] == 'success' else RED
                
                print(f"\n{status_color}╔══ EXECUTION ══╗{RESET}")
                print(f"  Status: {exec_res['final_status']}")
                
                if exec_res.get('stages', {}).get('vm_execution'):
                    vm = exec_res['stages']['vm_execution']
                    if vm.get('output'):
                        print(f"  Output: {vm['output'][:200]}...")
                
                print(f"{status_color}╚════════════════╝{RESET}")
            
            print(f"\n{PURPLE}Total time: {duration:.2f}s{RESET}")
        
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[*] Interrupted{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}[!] Error: {e}{RESET}")
    
    await agent.close()


if __name__ == "__main__":
    try:
        asyncio.run(interactive_shell())
    except Exception as e:
        print(f"{RED}Fatal error: {e}{RESET}")