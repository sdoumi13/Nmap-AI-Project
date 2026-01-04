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
    def __init__(self, model_checkpoint: str):
        print(f"{PURPLE}[*] Loading Diffusion Model...{RESET}")
        
        if not os.path.exists(model_checkpoint):
            self.model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
        else:
            self.model = NmapDiscreteDiffusionLM(model_name=model_checkpoint, use_adapter=False)
        
        self.sampler = DiscreteDiffusionSampler(self.model, max_steps=15)
        
        print(f"{GREEN}✓ Diffusion Agent (Generator) ready{RESET}")
    
    async def generate(self, user_query: str) -> str:
        try:
            print(f"{CYAN}[DIFFUSION] Generating command...{RESET}")
            start = time.time()
            
            result = self.sampler.sample(user_query, verbose=False)
            gen_time = time.time() - start
            
            command = result['final_command']
            print(f"  Generated: {command}")
            print(f"  Time: {gen_time:.2f}s")
            
            return command
        except Exception as e:
            print(f"{RED}[DIFFUSION] Generation error: {e}{RESET}")
            return None
    
    async def close(self):
        """Cleanup resources"""
        pass

async def interactive_shell():
    """
    Interactive shell for testing Diffusion Agent directly as a pure generator.
    NOTE: In production, use RouterAgent instead (agent_1_router/run_router.py)
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{PURPLE}{BOLD}{'='*70}")
    print("      DIFFUSION AGENT v3.2 (Pure Generator)")
    print("      ⚠️  FOR TESTING ONLY - Use RouterAgent in production")
    print(f"{'='*70}{RESET}\n")
    
    try:
        agent = DiffusionAgent(model_checkpoint="nmap_diffusion_checkpoint")
    except Exception as e:
        print(f"{RED}Init error: {e}{RESET}")
        return
    
    print("-" * 70)
    print("This is a GENERATOR ONLY - no decision-making, validation, or execution")
    print("-" * 70)
    
    while True:
        try:
            user_input = input(f"\n{PURPLE}DIFFUSION GENERATOR > {RESET}")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
            
            print(f"\n{YELLOW}Generating command...{RESET}\n")
            command = await agent.generate(user_input)
            
            if command:
                print(f"\n{GREEN}╔══ GENERATED COMMAND ══╗{RESET}")
                print(f"  {command}")
                print(f"{GREEN}╚══════════════════════════╝{RESET}")
                print(f"\n💡 In production, this would be sent to MCP Agent 5 for validation & execution")
            else:
                print(f"{RED}Failed to generate command{RESET}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
    
    await agent.close()

if __name__ == "__main__":
    asyncio.run(interactive_shell())