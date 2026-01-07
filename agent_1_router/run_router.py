# Fichier: agent_1_router/run_router.py
"""
RouterAgent - Central Orchestrator with CORS FIXED
User Query → Complexity → Distributed RAG or Diffusion → MCP Agent 5 → Validation + Execution
"""

import sys
import os
import asyncio
import httpx
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

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
MCP_AGENT5_URL = "http://localhost:5002"  # UPDATED PORT

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
                "relevant": False,
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
                "relevant": True,
                "agent": agent_choice,
                "analysis": complexity_result
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
        
        # Display execution report
        self._display_mcp_report(mcp_result)
        
        return {
            "status": "executed",
            "relevant": True,
            "complexity": complexity_result,
            "analysis": complexity_result,
            "agent": agent_choice,
            "command_generated": command,
            "generated_command": command,
            "generation_method": agent_choice,
            "entry_id": f"entry_{int(asyncio.get_event_loop().time())}",
            "execution": mcp_result
        }
    
    def _display_mcp_report(self, mcp_result: Dict[str, Any]):
        """Display MCP execution report"""
        report = mcp_result.get('stages', {})
        
        print(f"\n{YELLOW}{'─'*66}{RESET}")
        print(f"{YELLOW}MCP EXECUTION REPORT{RESET}")
        print(f"{YELLOW}{'─'*66}{RESET}")
        
        # Display stages...
        # (Keep existing display logic from original file)
        
        final_status = mcp_result.get('final_status', 'unknown')
        status_color = GREEN if final_status == 'success' else RED
        print(f"\n{YELLOW}{'─'*66}{RESET}")
        print(f"{status_color}{'✅ EXECUTION COMPLETED' if final_status == 'success' else '❌ EXECUTION FAILED'}{RESET}")
        print(f"{YELLOW}{'─'*66}{RESET}\n")
    
    async def _generate_rag_command(self, query: str, target: str) -> str:
        """Generate command using colleague's RAG Agent"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from agent_1_router.distributed_routing import DistributedRAGClient
            
            print(f"  {YELLOW}[Distributed Mode] Sending to colleague RAG...{RESET}")
            
            client = DistributedRAGClient(rag_url="http://192.168.1.218:8000")
            result = await client.generate_command(query=query, target=target)
            
            if result.get('status') == 'success':
                command = result.get('command')
                print(f"  {GREEN}[Colleague RAG] ✅ Command received: {command}{RESET}")
                command = self._ensure_target_in_command(command, target)
                return command
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"  ❌ Colleague RAG Error: {error_msg}")
                return None
        except Exception as e:
            print(f"  ❌ Distributed RAG Exception: {e}")
            print(f"  {YELLOW}[Fallback] Using basic nmap command...{RESET}")
            return f"nmap -sV {target}"
    
    async def _generate_diffusion_command(self, query: str, target: str) -> str:
        """Generate command using Diffusion Agent"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "diffusion_models"))
            from discrete_diffusion_nmap import NmapDiscreteDiffusionLM, DiscreteDiffusionSampler
            
            print(f"  {YELLOW}[Diffusion] Generating command...{RESET}")
            
            model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
            sampler = DiscreteDiffusionSampler(model, max_steps=15)
            
            result = sampler.sample(query, verbose=False)
            command = result['final_command']
            
            command = self._enhance_command_with_intent(command, query)
            command = self._ensure_target_in_command(command, target)
            
            print(f"  {GREEN}[Diffusion] ✅ Generated: {command}{RESET}")
            return command
            
        except Exception as e:
            print(f"  ❌ Diffusion Error: {e}")
            return None
    
    async def _validate_and_enhance_command(self, command: str, intent: str, target: str) -> str:
        """Validate and enhance command"""
        if not self.validator:
            return command
        
        try:
            print(f"\n  {CYAN}[Pre-Validation] Checking command quality...{RESET}")
            result = await self.validator.validate(command=command, intent=intent, agent_name="router")
            
            score = result.final_score
            print(f"    Validation Score: {score}/100")
            
            if score < 80:
                enhanced = self._enhance_command_with_intent(command, intent)
                if enhanced != command:
                    command = enhanced
                    print(f"    Enhanced: {command}")
            
            return command
        except Exception as e:
            print(f"    ⚠️ Validation skipped: {str(e)[:50]}")
            return command
    
    def _ensure_target_in_command(self, command: str, target: str) -> str:
        """Ensure command includes target IP"""
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
    
    def _enhance_command_with_intent(self, command: str, intent: str) -> str:
        """Enhance command based on intent keywords"""
        if not command or not intent:
            return command
        
        intent_lower = intent.lower()
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
            'vulnerability': '--script vuln',
        }
        
        for keyword, flag in intent_flags.items():
            if keyword in intent_lower and flag not in command:
                if command.strip().startswith('nmap'):
                    command = command.replace('nmap', f'nmap {flag}', 1)
        
        return command.strip()
    
    async def close(self):
        await self.mcp_client.close()


# ============================================================================
# FASTAPI APPLICATION WITH CORS
# ============================================================================

app = FastAPI(title="Router Agent", version="1.0.0")

# CRITICAL: Add CORS middleware
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

router_agent = None

@app.on_event("startup")
async def startup_event():
    global router_agent
    print("\n" + "="*70)
    print("🚀 Starting Router Agent...")
    print("="*70)
    router_agent = RouterAgent()


@app.get("/")
async def root():
    return {"service": "Router Agent", "status": "online"}


@app.post("/route")
async def route_endpoint(request: Request):
    """Main routing endpoint"""
    if router_agent is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Router agent not initialized"}
        )
    
    try:
        data = await request.json()
        user_query = data.get("query")
        target = data.get("target", "192.168.188.128")
        
        print(f"\n📥 Received routing request:")
        print(f"   Query: {user_query}")
        print(f"   Target: {target}")
        
        result = await router_agent.route(user_query, target)
        return result
        
    except Exception as e:
        print(f"❌ Route error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   ROUTER AGENT WITH CORS                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")