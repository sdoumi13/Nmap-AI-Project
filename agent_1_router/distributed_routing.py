"""
Distributed Routing Module
Handles routing between local agents and colleague's agents
"""

import httpx
from typing import Dict, Any
from datetime import datetime

# Colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


class DistributedRAGClient:
    """
    Client for remote RAG Agent (Colleague's machine)
    Handles communication with 192.168.1.218:8000
    """
    
    def __init__(self, rag_url: str = "http://192.168.1.218:8000"):
        self.base_url = rag_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.colleague_ip = "192.168.1.218"
    
    async def generate_command(self, query: str, target: str) -> Dict[str, Any]:
        """
        Request command generation from colleague's RAG
        
        Args:
            query: User query (e.g., "scan network for open ports")
            target: Target IP/network
            
        Returns:
            Dict with command, intent, confidence, etc.
        """
        try:
            print(f"{YELLOW}[DistributedRAG] Requesting from {self.colleague_ip}:8000...{RESET}")
            
            response = await self.client.post(
                f"{self.base_url}/generate_command",
                json={
                    "query": query,
                    "target": target,
                    "timestamp": datetime.now().isoformat()
                },
                headers={"X-Request-Source": "agent1-router"}
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"{GREEN}[DistributedRAG] ✅ Command received: {result.get('command')}{RESET}")
            
            return result
            
        except httpx.HTTPError as e:
            error_msg = f"Failed to reach colleague RAG at {self.colleague_ip}:8000 - {str(e)}"
            print(f"{RED}[DistributedRAG] ❌ {error_msg}{RESET}")
            
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"{RED}[DistributedRAG] ❌ Unexpected error: {str(e)}{RESET}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def health_check(self) -> bool:
        """Check if colleague's RAG is available"""
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=5.0
            )
            is_healthy = response.status_code == 200
            status = "🟢 ONLINE" if is_healthy else "🔴 OFFLINE"
            print(f"{CYAN}[DistributedRAG] {status} - {self.colleague_ip}:8000{RESET}")
            return is_healthy
        except:
            print(f"{RED}[DistributedRAG] 🔴 OFFLINE - {self.colleague_ip}:8000{RESET}")
            return False
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class LocalDiffusionClient:
    """
    Client for local Diffusion Agent
    Handles complex command generation locally
    """
    
    def __init__(self, diffusion_url: str = "http://192.168.1.169:9000"):
        self.base_url = diffusion_url
        self.client = httpx.AsyncClient(timeout=90.0)
    
    async def generate_command(self, query: str, target: str) -> Dict[str, Any]:
        """Generate command using local Diffusion model"""
        try:
            print(f"{YELLOW}[LocalDiffusion] Generating complex command...{RESET}")
            
            response = await self.client.post(
                f"{self.base_url}/generate",
                json={
                    "query": query,
                    "target": target,
                    "complexity": "HIGH"
                }
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"{GREEN}[LocalDiffusion] ✅ Command: {result.get('command')}{RESET}")
            
            return result
            
        except httpx.HTTPError as e:
            error_msg = f"Diffusion generation failed: {str(e)}"
            print(f"{RED}[LocalDiffusion] ❌ {error_msg}{RESET}")
            
            return {
                "status": "error",
                "error": error_msg
            }
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class DistributedRouter:
    """
    Smart routing logic based on complexity classification
    Routes to colleague's RAG or local Diffusion
    """
    
    def __init__(self, 
                 rag_colleague_url: str = "http://192.168.1.218:8000",
                 diffusion_local_url: str = "http://192.168.1.169:9000"):
        
        self.rag_colleague = DistributedRAGClient(rag_colleague_url)
        self.diffusion_local = LocalDiffusionClient(diffusion_local_url)
    
    async def route_to_agent(self, query: str, target: str, 
                            complexity: str, confidence: float) -> Dict[str, Any]:
        """
        Route query to appropriate agent based on complexity
        
        SIMPLE/MEDIUM (confidence > 0.7)
        └─→ Colleague's RAG (192.168.1.218)
        
        COMPLEX
        └─→ Your Diffusion (192.168.1.169)
        """
        
        print(f"\n{CYAN}{'='*60}")
        print(f"ROUTING DECISION: {complexity} (confidence: {confidence:.0%})")
        print(f"{'='*60}{RESET}")
        
        if complexity in ["SIMPLE", "MEDIUM"] and confidence > 0.7:
            # Route to colleague's RAG
            print(f"{CYAN}→ Routing to COLLEAGUE'S RAG (192.168.1.218:8000){RESET}")
            
            result = await self.rag_colleague.generate_command(query, target)
            
            if result.get("status") == "error":
                print(f"{YELLOW}[Router] RAG failed, falling back to local Diffusion...{RESET}")
                result = await self.diffusion_local.generate_command(query, target)
            
            result["source_agent"] = "RAG_COLLEAGUE"
            result["source_ip"] = "192.168.1.218"
            
        else:
            # Route to local Diffusion
            print(f"{CYAN}→ Routing to YOUR DIFFUSION (192.168.1.169:9000){RESET}")
            
            result = await self.diffusion_local.generate_command(query, target)
            result["source_agent"] = "DIFFUSION_LOCAL"
            result["source_ip"] = "192.168.1.169"
        
        return result
    
    async def check_distributed_health(self) -> Dict[str, bool]:
        """Check health of all distributed agents"""
        print(f"\n{CYAN}[Router] Checking distributed agent health...{RESET}")
        
        health = {
            "colleague_rag": await self.rag_colleague.health_check(),
            "local_diffusion": True  # Assume local is available for now
        }
        
        return health
    
    async def close(self):
        """Close all client connections"""
        await self.rag_colleague.close()
        await self.diffusion_local.close()
