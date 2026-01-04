# Fichier: agent_1_router/run_router.py
"""
RouterAgent - Central Orchestrator
User Query → Complexity Decision → Agent (RAG or Diffusion) → MCP Agent 5 → Validation + Correction + Sandbox + VM
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

# URLs
COMPLEXITY_URL = "http://localhost:7000"
MCP_AGENT5_URL = "http://localhost:5000"
RAG_AGENT_URL = "http://localhost:8000"  # If RAG runs as a service
DIFFUSION_AGENT_URL = "http://localhost:9000"  # If Diffusion runs as a service

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
PURPLE = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ComplexityClient:
    """Client for Complexity Classification API"""
    def __init__(self, api_url: str):
        self.base_url = api_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def classify(self, query: str) -> Dict[str, Any]:
        try:
            response = await self.client.post(
                f"{self.base_url}/classify",
                json={"query": query, "user_id": "router-agent"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {
                "complexity": "MEDIUM",
                "confidence": 0.5,
                "recommended_agent": "DIFFUSION",
                "reasoning": f"API error: {str(e)}"
            }
    
    async def close(self):
        await self.client.aclose()


class MCPClient:
    """Client for MCP Server (Agent 5)"""
    def __init__(self, mcp_url: str):
        self.base_url = mcp_url
        self.client = httpx.AsyncClient(timeout=180.0)
    
    async def execute_command(self, command: str, intent: str, target: str, 
                            agent_name: str) -> Dict[str, Any]:
        """Call MCP execute endpoint (validation + correction + sandbox + VM)"""
        try:
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


class RouterAgent:
    """
    Central Router: Decides RAG or Diffusion, routes to appropriate agent, 
    then sends to MCP for execution
    """
    
    def __init__(self, complexity_url: str, mcp_url: str):
        print(f"{PURPLE}[*] Initializing RouterAgent...{RESET}")
        
        self.complexity_client = ComplexityClient(complexity_url)
        self.mcp_client = MCPClient(mcp_url)
        self.comp_agent = ComprehensionAgent()
        self.complexity_agent = ComplexityAgent()
        
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
        
        # STEP 4: MCP Execution (Agent 5 pipeline)
        print(f"\n{CYAN}[STEP 4/4] MCP EXECUTION (AGENT 5){RESET}")
        
        mcp_result = await self.mcp_client.execute_command(
            command=command,
            intent=user_query,
            target=target,
            agent_name=agent_choice.lower()
        )
        
        return {
            "status": "executed",
            "complexity": complexity_result,
            "agent": agent_choice,
            "command_generated": command,
            "execution": mcp_result
        }
    
    async def _generate_rag_command(self, query: str, target: str) -> str:
        """
        Generate command using RAG Agent.
        RAG is a pure generator - no decisions, only generation.
        """
        try:
            # Import RAG agent directly (not via HTTP)
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from RAG.agent.rag_agent import NmapRagAgent
            
            rag = NmapRagAgent()
            result = rag.process({
                "user_query": query,
                "extracted_ip": target
            })
            
            if result['status'] == 'success':
                return result['nmap_candidate']
            else:
                return None
        except Exception as e:
            print(f"  ❌ RAG Error: {e}")
            return None
    
    async def _generate_diffusion_command(self, query: str, target: str) -> str:
        """
        Generate command using Diffusion Agent.
        Diffusion is a pure generator - no decisions, only generation.
        """
        try:
            # Import Diffusion agent directly
            sys.path.insert(0, str(Path(__file__).parent.parent / "diffusion_models"))
            from discrete_diffusion_nmap import DiscreteDiffusionSampler, NmapDiscreteDiffusionLM
            
            model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
            sampler = DiscreteDiffusionSampler(model, max_steps=15)
            
            result = sampler.sample(query, verbose=False)
            return result['final_command']
        except Exception as e:
            print(f"  ❌ Diffusion Error: {e}")
            return None
    
    async def close(self):
        await self.complexity_client.close()
        await self.mcp_client.close()


async def main():
    """Interactive shell for RouterAgent"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{PURPLE}{BOLD}{'='*70}")
    print("      NMAP-AI ROUTER v1.0")
    print("      User Query → Complexity → Agent → MCP → Validation → Sandbox → VM")
    print(f"{'='*70}{RESET}\n")
    
    try:
        router = RouterAgent(
            complexity_url=COMPLEXITY_URL,
            mcp_url=MCP_AGENT5_URL
        )
    except Exception as e:
        print(f"{RED}Init error: {e}{RESET}")
        return
    
    print("-" * 70)
    print(f"Default Target: 192.168.188.128 (Ubuntu VM)")
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
                
                print(f"\n{GREEN}╔══ COMMAND ══╗{RESET}")
                print(f"  {result['command_generated']}")
                print(f"{GREEN}╚════════════════╝{RESET}")
                
                # Self-correction info
                if exec_res.get('stages', {}).get('self_correction', {}).get('applied'):
                    corr = exec_res['stages']['self_correction']
                    print(f"\n{YELLOW}╔══ CORRECTED ══╗{RESET}")
                    print(f"  {corr['final_command']}")
                    print(f"  Score: {corr.get('final_score', 'N/A')}/100")
                    print(f"{YELLOW}╚════════════════╝{RESET}")
                
                # Final status
                status = exec_res.get('final_status', 'unknown')
                color = GREEN if status == 'success' else RED
                
                print(f"\n{color}╔══ FINAL STATUS ══╗{RESET}")
                print(f"  {status}")
                if status != 'success':
                    errors = exec_res.get('stages', {}).get('vm_execution', {}).get('errors', [])
                    for err in errors:
                        print(f"  Error: {err}")
                print(f"{color}╚═════════════════╝{RESET}")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
    
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())