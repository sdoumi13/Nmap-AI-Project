"""
agent3_diffusion_with_mcp.py
Diffusion Agent → Complexity API → MCP Server (Agent 5)
Cible VM Ubuntu: 192.168.188.128
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
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
PURPLE = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

class ComplexityClient:
    """Client pour Complexity API"""
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
                "reasoning": f"API error: {str(e)}"
            }
    
    async def close(self):
        await self.client.aclose()

class MCPClient:
    """Client pour MCP Server (Agent 5)"""
    def __init__(self, mcp_url: str):
        self.base_url = mcp_url
        self.client = httpx.AsyncClient(timeout=180.0)
    
    async def execute_command(self, command: str, intent: str, target: str, 
                            agent_name: str = "diffusion-agent") -> Dict[str, Any]:
        """Appel MCP execute endpoint (validation + correction + sandbox + VM)"""
        try:
            # Envoi vers le pipeline Agent 5
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

class DiffusionAgent:
    """Diffusion Agent avec Complexity + MCP"""
    
    def __init__(self, model_checkpoint: str, complexity_url: str, mcp_url: str):
        print(f"{PURPLE}[*] Loading Diffusion Model...{RESET}")
        
        if not os.path.exists(model_checkpoint):
            self.model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
        else:
            self.model = NmapDiscreteDiffusionLM(model_name=model_checkpoint, use_adapter=False)
        
        self.sampler = DiscreteDiffusionSampler(self.model, max_steps=15)
        self.complexity_client = ComplexityClient(complexity_url)
        self.mcp_client = MCPClient(mcp_url)
        
        print(f"{GREEN}✓ Diffusion + Complexity + MCP ready{RESET}")
    
    async def process_query(self, user_query: str, target: str, force_diffusion: bool = False) -> Dict[str, Any]:
        """Pipeline: Complexity → Diffusion → MCP (Agent 5)"""
        
        # STEP 1: Complexity
        print(f"{CYAN}[1/3] Complexity Analysis{RESET}")
        complexity = await self.complexity_client.classify(user_query)
        
        print(f"  Level: {complexity['complexity']}")
        print(f"  Confidence: {complexity['confidence']:.2f}")
        print(f"  Recommended: {complexity['recommended_agent']}\n")
        
        if not force_diffusion and complexity['recommended_agent'] != 'DIFFUSION':
            return {
                "status": "routed",
                "complexity": complexity,
                "message": f"Query routed to {complexity['recommended_agent']}"
            }
        
        # STEP 2: Diffusion Generation
        print(f"{CYAN}[2/3] Diffusion Generation{RESET}")
        
        start = time.time()
        diff_result = self.sampler.sample(user_query, verbose=False)
        gen_time = time.time() - start
        
        command = diff_result['final_command']
        print(f"  Command: {command}")
        print(f"  Time: {gen_time:.2f}s\n")
        
        # STEP 3: MCP Execution (Agent 5 pipeline)
        print(f"{CYAN}[3/3] MCP Execution (Agent 5){RESET}")
        
        # Remplacement manuel de "TARGET" par l'IP réelle si nécessaire avant envoi
        final_cmd = command.replace("TARGET", target)
        mcp_result = await self.mcp_client.execute_command(final_cmd, user_query, target, "diffusion-agent")
        
        return {
            "status": "executed",
            "complexity": complexity,
            "command": final_cmd,
            "diffusion_time": gen_time,
            "execution": mcp_result
        }
    
    async def close(self):
        await self.complexity_client.close()
        await self.mcp_client.close()

async def interactive_shell():
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{PURPLE}{BOLD}{'='*70}")
    print("      DIFFUSION AGENT v3.1")
    print("      Query → Complexity API → Diffusion → Agent 5 (VM Ubuntu)")
    print(f"{'='*70}{RESET}\n")
    
    try:
        agent = DiffusionAgent(
            model_checkpoint="nmap_diffusion_checkpoint",
            complexity_url=COMPLEXITY_API_URL,
            mcp_url=MCP_AGENT5_URL
        )
    except Exception as e:
        print(f"{RED}Init error: {e}{RESET}")
        return
    
    print("-" * 70)
    print(f"Default Target: 192.168.188.128 (Ubuntu VM)")
    print("-" * 70)
    
    force_diffusion = False
    
    while True:
        try:
            user_input = input(f"\n{PURPLE}DIFFUSION {'[FORCED]' if force_diffusion else ''} > {RESET}")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            if user_input.lower() == 'force':
                force_diffusion = not force_diffusion
                print(f"{YELLOW}Force: {'ON' if force_diffusion else 'OFF'}{RESET}")
                continue
            if not user_input.strip():
                continue
            
            # Modification: IP de la VM par défaut
            target = input(f"{YELLOW}Target (Default 192.168.188.128): {RESET}") or "192.168.188.128"
            
            print(f"\n{YELLOW}Processing...{RESET}\n")
            result = await agent.process_query(user_input, target, force_diffusion)
            
            # Affichage des résultats du pipeline Agent 5
            if result["status"] == "routed":
                print(f"{CYAN}╔══ ROUTED ══╗{RESET}")
                print(f"  {result['message']}")
                print(f"{CYAN}╚═════════════╝{RESET}")
            else:
                exec_res = result["execution"]
                
                print(f"{GREEN}╔══ COMMAND ══╗{RESET}")
                print(f"  {result['command']}")
                print(f"{GREEN}╚════════════════╝{RESET}")
                
                # Gestion de l'affichage de l'auto-correction
                if exec_res.get('stages', {}).get('self_correction', {}).get('applied'):
                    corr = exec_res['stages']['self_correction']
                    print(f"\n{YELLOW}╔══ CORRECTED ══╗{RESET}")
                    print(f"  {corr['final_command']}")
                    print(f"  Final Validation Score: {corr.get('final_validation_score', 'N/A')}/100")
                    print(f"{YELLOW}╚════════════════╝{RESET}")
                
                # Statut Final de l'exécution VM
                status = exec_res.get('final_status', 'unknown')
                color = GREEN if status == 'success' else RED
                
                print(f"\n{color}╔══ VM STATUS ══╗{RESET}")
                print(f"  {status}")
                if status != 'success':
                    # Affiche les erreurs SSH ou Nmap si disponibles
                    errors = exec_res.get('stages', {}).get('vm_execution', {}).get('errors', [])
                    for err in errors:
                        print(f"  Error: {err}")
                print(f"{color}╚════════════════╝{RESET}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
    
    await agent.close()

if __name__ == "__main__":
    asyncio.run(interactive_shell())