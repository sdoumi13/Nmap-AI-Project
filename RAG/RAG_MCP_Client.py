"""
Agent 2 - RAG avec intégration MCP Client + Complexity Routing
Architecture: Query → Complexity API → RAG Generation → MCP Validation → Agent 5
"""

import os
import sys
import time
import asyncio
import httpx
from typing import Dict, Any, Optional
from agent.rag_agent import NmapRagAgent

# URLs
COMPLEXITY_API_URL = "http://localhost:7000"  # Agent 1 Complexity Classifier
MCP_AGENT5_URL = "http://localhost:5000"       # Agent 5 MCP Server

# Couleurs terminal
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ComplexityClient:
    """Client pour interroger l'API de classification de complexité"""
    
    def __init__(self, api_url: str):
        self.base_url = api_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifie la complexité de la requête
        
        Returns:
            {
                "complexity": "EASY"|"MEDIUM"|"HARD",
                "confidence": float,
                "recommended_agent": "RAG"|"DIFFUSION",
                "reasoning": str
            }
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/classify",
                json={
                    "query": query,
                    "user_id": "rag-agent"
                }
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPError as e:
            return {
                "complexity": "UNKNOWN",
                "confidence": 0.0,
                "recommended_agent": "RAG",  # Fallback
                "reasoning": f"Complexity API error: {str(e)}"
            }
    
    async def close(self):
        await self.client.aclose()


class MCPClient:
    """Client MCP pour communiquer avec Agent 5"""
    
    def __init__(self, agent5_url: str):
        self.base_url = agent5_url
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def validate_command(
        self, 
        command: str, 
        intent: str, 
        agent_name: str = "rag-agent"
    ) -> Dict[str, Any]:
        """
        Envoie une commande à Agent 5 pour validation via MCP
        
        Returns:
            {
                "valid": bool,
                "status": str,
                "score": int,
                "errors": list,
                "warnings": list,
                "method_used": str
            }
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/mcp/validate",
                json={
                    "command": command,
                    "intent": intent,
                    "agent_name": agent_name,
                    "context": {
                        "source": "rag",
                        "retrieval_score": 0.85
                    }
                }
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPError as e:
            return {
                "valid": False,
                "status": "error",
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
        agent_name: str = "rag-agent"
    ) -> Dict[str, Any]:
        """
        Envoie une commande complète pour validation + exécution
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/mcp/execute",
                json={
                    "command": command,
                    "intent": intent,
                    "target": target,
                    "agent_name": agent_name
                }
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPError as e:
            return {
                "final_status": "mcp_error",
                "error": str(e)
            }
    
    async def close(self):
        await self.client.aclose()


class RagAgentWithMCP:
    """RAG Agent avec intégration Complexity Routing + MCP"""
    
    def __init__(
        self, 
        dataset_path: str, 
        complexity_url: str,
        mcp_url: str
    ):
        self.rag_agent = NmapRagAgent(dataset_path=dataset_path)
        self.complexity_client = ComplexityClient(complexity_url)
        self.mcp_client = MCPClient(mcp_url)
    
    async def process_with_routing(
        self, 
        user_query: str,
        validate_only: bool = True,
        force_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Pipeline complet: Complexity Check → Generation → MCP Validation
        
        Args:
            user_query: Requête utilisateur
            validate_only: Si True, valide seulement. Si False, exécute aussi.
            force_rag: Force l'utilisation de RAG (ignore routing)
        
        Returns:
            Résultat complet avec routing + génération + validation
        """
        
        # STEP 1: Complexity Classification
        print(f"{YELLOW}[1/3] Analyzing query complexity...{RESET}")
        complexity_result = await self.complexity_client.classify(user_query)
        
        print(f"\n{BLUE}╔══ COMPLEXITY ANALYSIS ══╗{RESET}")
        print(f"  Level: {complexity_result['complexity']}")
        print(f"  Confidence: {complexity_result['confidence']:.2f}")
        print(f"  Recommended: {complexity_result['recommended_agent']}")
        print(f"  Reason: {complexity_result['reasoning']}")
        print(f"{BLUE}╚═════════════════════════╝{RESET}\n")
        
        # Check if RAG should handle this
        if not force_rag and complexity_result['recommended_agent'] != 'RAG':
            return {
                "status": "routed_to_diffusion",
                "complexity": complexity_result,
                "message": f"Query routed to DIFFUSION agent due to {complexity_result['complexity']} complexity",
                "command": None,
                "validation": None
            }
        
        # STEP 2: RAG Generation
        print(f"{YELLOW}[2/3] Generating command via RAG...{RESET}")
        rag_result = self.rag_agent.process({"user_query": user_query})
        
        if rag_result.get("status") != "success":
            return {
                "status": "rag_failed",
                "complexity": complexity_result,
                "rag_error": rag_result.get("error_message"),
                "command": None,
                "validation": None
            }
        
        command = rag_result.get("nmap_candidate")
        
        print(f"{GREEN}✓ Command generated: {command}{RESET}\n")
        
        # STEP 3: MCP Validation
        print(f"{YELLOW}[3/3] Validating via MCP (Agent 5)...{RESET}")
        validation = await self.mcp_client.validate_command(
            command=command,
            intent=user_query,
            agent_name="rag-agent"
        )
        
        result = {
            "status": "validated",
            "complexity": complexity_result,
            "command": command,
            "rag_context": rag_result.get("context", []),
            "validation": validation
        }
        
        # STEP 4: Execution optionnelle
        if not validate_only and validation.get("valid"):
            target = input(f"{YELLOW}Enter target (e.g., scanme.nmap.org): {RESET}")
            
            execution = await self.mcp_client.execute_command(
                command=command,
                intent=user_query,
                target=target,
                agent_name="rag-agent"
            )
            
            result["execution"] = execution
            result["status"] = "executed"
        
        return result
    
    async def close(self):
        await self.complexity_client.close()
        await self.mcp_client.close()


async def interactive_shell():
    """Shell interactif avec Complexity Routing + MCP"""
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{BLUE}{BOLD}" + "="*70)
    print("      NMAP RAG AGENT v3.0 - COMPLEXITY ROUTING + MCP")
    print("      Architecture: Query → Complexity → RAG → MCP → Agent 5")
    print("="*70 + f"{RESET}\n")
    
    # Vérifications
    if not os.path.exists("nmap_dataset.json"):
        print(f"{RED}[!] Error: nmap_dataset.json not found!{RESET}")
        return
    
    print(f"{YELLOW}[*] Initializing RAG + Complexity + MCP...{RESET}")
    
    try:
        agent = RagAgentWithMCP(
            dataset_path="nmap_dataset.json",
            complexity_url=COMPLEXITY_API_URL,
            mcp_url=MCP_AGENT5_URL
        )
        print(f"{GREEN}[✓] Agent RAG + MCP ready!{RESET}")
    except Exception as e:
        print(f"{RED}[✗] Initialization error: {e}{RESET}")
        return
    
    print("-" * 70)
    print("Available commands:")
    print("  • Type your query in natural language")
    print("  • 'validate' : Validation mode only")
    print("  • 'execute' : Validation + execution mode")
    print("  • 'force-rag' : Force RAG (ignore routing)")
    print("  • 'exit' / 'quit' : Exit")
    print("-" * 70)
    
    mode = "validate"
    force_rag = False
    
    while True:
        try:
            user_input = input(f"\n{BLUE}{BOLD}RAG [{mode}{'|FORCED' if force_rag else ''}] > {RESET}")
            
            # System commands
            if user_input.lower() in ['exit', 'quit']:
                print(f"{YELLOW}[*] Closing...{RESET}")
                break
            
            if user_input.lower() == 'validate':
                mode = "validate"
                print(f"{GREEN}Mode: Validation only{RESET}")
                continue
            
            if user_input.lower() == 'execute':
                mode = "execute"
                print(f"{YELLOW}Mode: Validation + Execution{RESET}")
                continue
            
            if user_input.lower() == 'force-rag':
                force_rag = not force_rag
                print(f"{YELLOW}Force RAG: {'ON' if force_rag else 'OFF'}{RESET}")
                continue
            
            if not user_input.strip():
                continue
            
            # Process
            print(f"{YELLOW}[...] Processing...{RESET}\n")
            start_time = time.time()
            
            result = await agent.process_with_routing(
                user_query=user_input,
                validate_only=(mode == "validate"),
                force_rag=force_rag
            )
            
            duration = time.time() - start_time
            
            # Display results
            if result["status"] == "routed_to_diffusion":
                print(f"\n{YELLOW}╔══ ROUTED TO DIFFUSION ══╗{RESET}")
                print(f"  {result['message']}")
                print(f"{YELLOW}╚═════════════════════════╝{RESET}")
                print(f"\n{RED}Please use the Diffusion Agent for this query.{RESET}")
                continue
            
            if result["status"] == "rag_failed":
                print(f"{RED}[✗] RAG failed: {result['rag_error']}{RESET}")
                continue
            
            # Generated command
            print(f"\n{GREEN}╔══ GENERATED COMMAND ══╗{RESET}")
            print(f"{BOLD}  {result['command']}{RESET}")
            print(f"{GREEN}╚═══════════════════════╝{RESET}")
            
            # Validation
            validation = result["validation"]
            status_color = GREEN if validation["valid"] else RED
            
            print(f"\n{status_color}╔══ MCP VALIDATION ══╗{RESET}")
            print(f"  Status: {validation['status']}")
            print(f"  Score: {validation['score']}/100")
            print(f"  Method: {validation['method_used']}")
            
            if validation.get("errors"):
                print(f"{RED}  Errors:{RESET}")
                for err in validation["errors"]:
                    print(f"    • {err}")
            
            if validation.get("warnings"):
                print(f"{YELLOW}  Warnings:{RESET}")
                for warn in validation["warnings"]:
                    print(f"    • {warn}")
            
            print(f"{status_color}╚════════════════════╝{RESET}")
            
            # Execution (if requested)
            if result.get("execution"):
                exec_result = result["execution"]
                
                if exec_result.get("final_status") == "success":
                    print(f"\n{GREEN}╔══ EXECUTION SUCCESS ══╗{RESET}")
                    vm_output = exec_result["stages"]["vm_execution"]["output"]
                    print(f"{vm_output[:500]}...")
                    print(f"{GREEN}╚═══════════════════════╝{RESET}")
                else:
                    print(f"\n{RED}╔══ EXECUTION FAILED ══╗{RESET}")
                    print(f"  Status: {exec_result['final_status']}")
                    print(f"{RED}╚═══════════════════════╝{RESET}")
            
            print(f"\n{BLUE}Total time: {duration:.2f}s{RESET}")
        
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[*] Interrupt detected{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}[!] Error: {e}{RESET}")
    
    # Cleanup
    await agent.close()


if __name__ == "__main__":
    try:
        asyncio.run(interactive_shell())
    except Exception as e:
        print(f"{RED}Fatal error: {e}{RESET}")
        import traceback
        traceback.print_exc()