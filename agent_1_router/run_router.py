"""
RouterAgent - Central Orchestrator 
JSON Error Fix + Proper CORS + Lifespan Events
"""

import sys
import os
import asyncio
import httpx
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_1_router.comprehension import ComprehensionAgent
from agent_1_router.complexity import ComplexityAgent
from agent_1_router.distributed_routing import DistributedRAGClient, FineTuningClient

# Import Hybrid Validator
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_5_validation"))
try:
    from agent_5_validation.validation.hybrid_validator import AdvancedHybridValidator
    HYBRID_VALIDATOR_AVAILABLE = True
except ImportError:
    HYBRID_VALIDATOR_AVAILABLE = False

# Configuration
COLLEAGUE_RAG_URL = "http://192.168.1.141:8000"
FINETUNING_URL = "https://82c0264a817e.ngrok-free.app"
MCP_AGENT5_URL = "http://localhost:5002"

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
PURPLE = "\033[95m"
RESET = "\033[0m"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class RouteRequest(BaseModel):
    query: str
    target: str = "192.168.188.128"


# ============================================================================
# MCP CLIENT
# ============================================================================

class MCPClient:
    """Client for MCP Server (Agent 5)"""
    def __init__(self, mcp_url: str):
        self.base_url = mcp_url
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def execute_command(self, command: str, intent: str, target: str, 
                            agent_name: str) -> Dict[str, Any]:
        """Call MCP execute endpoint"""
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
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"  {RED}[MCP] ✗ {error_msg}{RESET}")
            return {
                "final_status": "mcp_error",
                "command": command,
                "stages": {"error": error_msg},
                "timestamp": ""
            }
    
    async def close(self):
        await self.client.aclose()


# ============================================================================
# ROUTER AGENT
# ============================================================================

class RouterAgent:
    """Central Router with SLM Integration"""
    
    def __init__(self):
        print(f"{PURPLE}[*] Initializing RouterAgent...{RESET}")
        
        self.mcp_client = MCPClient(MCP_AGENT5_URL)
        self.comp_agent = ComprehensionAgent()
        self.complexity_agent = ComplexityAgent()
        
        if HYBRID_VALIDATOR_AVAILABLE:
            try:
                self.validator = AdvancedHybridValidator(
                    mistral_api_url="http://192.168.11.1:1235/v1/chat/completions"  # Port 1235 for Mistral
                )
                print(f"{GREEN}✓ Hybrid Validator initialized (Mistral on port 1235){RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠ Hybrid Validator unavailable: {e}{RESET}")
                self.validator = None
        else:
            self.validator = None
        
        print(f"{GREEN}✓ RouterAgent ready{RESET}")
    
    async def route(self, user_query: str, target: str) -> Dict[str, Any]:
        """Main routing logic"""
        
        print(f"\n{CYAN}{'='*70}")
        print(f"ROUTER PIPELINE")
        print(f"{'='*70}{RESET}")
        
        # STEP 1: Comprehension Check
        print(f"\n{CYAN}[STEP 1/4] COMPREHENSION CHECK{RESET}")
        comp_result = await self.comp_agent.analyze(user_query)
        
        if not comp_result['relevant']:
            print(f"  ✗ REJECTED. Score: {comp_result['score']:.2f}")
            print(f"     Reason: {comp_result['reason']}")
            return {
                "status": "rejected",
                "relevant": False,
                "reason": comp_result['reason'],
                "score": comp_result['score']
            }
        
        print(f"  ✅ VALID. Score: {comp_result['score']:.2f}")
        
        # STEP 2: Complexity Classification
        print(f"\n{CYAN}[STEP 2/4] COMPLEXITY CLASSIFICATION{RESET}")
        complexity_result = await self.complexity_agent.classify(user_query)
        
        level = complexity_result['level']
        level_color = GREEN if level == 'Easy' else YELLOW if level == 'Medium' else RED
        
        if level == "Easy":
            agent_choice = "RAG"
        elif level == "Medium":
            agent_choice = "FINETUNING"
        else:
            agent_choice = "DIFFUSION"
        
        print(f"  Level: {level_color}{level}{RESET}")
        print(f"  Confidence: {complexity_result['confidence']:.2f}")
        print(f"  Recommended Agent: {level_color}{agent_choice}{RESET}")
        
        # STEP 3: Command Generation
        print(f"\n{CYAN}[STEP 3/4] COMMAND GENERATION ({agent_choice}){RESET}")
        
        if agent_choice == "RAG":
            command = await self._generate_rag_command(user_query, target)
        elif agent_choice == "FINETUNING":
            command = await self._generate_finetuning_command(user_query, target)
        else:
            command = await self._generate_diffusion_command(user_query, target)
        
        if not command:
            return {
                "status": "generation_failed",
                "relevant": True,
                "agent": agent_choice,
                "analysis": complexity_result
            }
        
        print(f"  Generated: {command}")
        
        # STEP 4: MCP Execution
        print(f"\n{CYAN}[STEP 4/4] MCP EXECUTION (AGENT 5){RESET}")
        
        mcp_result = await self.mcp_client.execute_command(
            command=command,
            intent=user_query,
            target=target,
            agent_name=agent_choice.lower()
        )
        
        return {
            "status": "executed",
            "relevant": True,
            "complexity": complexity_result,
            "agent": agent_choice,
            "command_generated": command,
            "execution": mcp_result
        }
    
    async def _generate_rag_command(self, query: str, target: str) -> str:
        """Generate command using RAG"""
        try:
            client = DistributedRAGClient(rag_url=COLLEAGUE_RAG_URL)
            result = await client.generate_command(query=query, target=target)
            
            if result.get('status') == 'success':
                command = result.get('command')
                print(f"  {GREEN}[RAG] ✅ {command}{RESET}")
                return self._ensure_target_in_command(command, target)
            else:
                print(f"  ✗ RAG Error: {result.get('error')}")
                return None
        except Exception as e:
            print(f"  ✗ RAG Exception: {e}")
            return f"nmap -sV {target}"
    
    async def _generate_finetuning_command(self, query: str, target: str) -> str:
        """Generate command using Fine-Tuning"""
        try:
            client = FineTuningClient(finetuning_url=FINETUNING_URL)
            result = await client.generate_command(query=query, target=target)
            
            if result.get('status') == 'success':
                command = result.get('command')
                print(f"  {GREEN}[FineTuning] ✅ {command}{RESET}")
                return self._ensure_target_in_command(command, target)
            else:
                print(f"  ✗ FineTuning Error: {result.get('error')}")
                return None
        except Exception as e:
            print(f"  ✗ FineTuning Exception: {e}")
            return f"nmap -sV {target}"
    
    async def _generate_diffusion_command(self, query: str, target: str) -> str:
        """Generate command using Diffusion"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "diffusion_models"))
            from discrete_diffusion_nmap import NmapDiscreteDiffusionLM, DiscreteDiffusionSampler
            
            model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
            sampler = DiscreteDiffusionSampler(model, max_steps=15)
            
            result = sampler.sample(query, verbose=False)
            command = result['final_command']
            
            command = self._ensure_target_in_command(command, target)
            print(f"  {GREEN}[Diffusion] ✅ {command}{RESET}")
            return command
            
        except Exception as e:
            print(f"  ✗ Diffusion Error: {e}")
            return None
    
    def _ensure_target_in_command(self, command: str, target: str) -> str:
        """Ensure target is in command"""
        import re
        
        if not command:
            return command
        
        command = command.replace('<target>', target)
        command = command.replace('<TARGET>', target)
        command = command.replace('TARGET', target)
        
        ip_pattern = r'\d+\.\d+\.\d+\.\d+'
        has_ip = re.search(ip_pattern, command)
        
        if not has_ip and command.strip().startswith('nmap'):
            command = f"{command.strip()} {target}"
        
        return command.strip()
    
    async def close(self):
        await self.mcp_client.close()
        await self.comp_agent.close()
        await self.complexity_agent.close()


# ============================================================================
# FASTAPI APPLICATION WITH LIFESPAN
# ============================================================================

router_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager (replaces on_event)"""
    global router_agent
    
    # Startup
    print("\n" + "="*70)
    print("🚀 Starting Router Agent...")
    print("="*70)
    router_agent = RouterAgent()
    
    yield
    
    # Shutdown
    if router_agent:
        await router_agent.close()
    print("\n🛑 Router Agent stopped")


app = FastAPI(
    title="Router Agent",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "Router Agent", "status": "online"}


@app.get("/health")
async def health():
    return {
        "status": "online" if router_agent else "initializing",
        "slm_port": 1234,
        "agent5_port": 5002
    }


@app.post("/route")
async def route_endpoint(request: RouteRequest):
    """Main routing endpoint - FIXED JSON parsing"""
    if router_agent is None:
        raise HTTPException(status_code=503, detail="Router agent not initialized")
    
    try:
        print(f"\n🔥 Received routing request:")
        print(f"   Query: {request.query}")
        print(f"   Target: {request.target}")
        
        result = await router_agent.route(request.query, request.target)
        return result
        
    except Exception as e:
        print(f"✗ Route error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   ROUTER AGENT WITH CORS                      ║
    ║           Qwen2.5-Coder-3B on Port 1234 (Agent 1)             ║
    ║           Mistral-7B on Port 1235 (Agent 5)                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")