"""
MCP Server (Vrai SDK Anthropic)
Expose les outils de validation/exécution via MCP protocol
"""
# Fichier: mcp_tools/mcp_server.py
import json
import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route
from sse_starlette.sse import EventSourceResponse

# Import de VOS composants existants
from validation.hybrid_validator import AdvancedHybridValidator, ValidationStatus
from execution.sandbox_executor import SandboxExecutor
from execution.vm_executor import VMExecutor

class MCPServer:  # Renommé pour correspondre à votre import dans run_agent5.py
    """
    Serveur MCP Officiel pour Agent 5.
    Expose les outils de validation et d'exécution via le protocole MCP/SSE.
    """
    
    def __init__(self, config: dict = None):
        # Initialisation du serveur MCP
        self.server = Server("nmap-ai-agent5")
        
        # Chargement de la config (gestion du cas où config est None pour les tests)
        self.config = config if config else {}
        api_url = self.config.get('llm', {}).get('api_url', "http://192.168.11.1:1234/v1/chat/completions")
        vm_config = self.config.get('vm', {})

        # Initialisation de vos moteurs (Logic)
        print("🔌 Initialisation des moteurs MCP...")
        self.validator = AdvancedHybridValidator(mistral_api_url=api_url)
        self.sandbox = SandboxExecutor()
        self.vm = VMExecutor(vm_config)
        
        # Enregistrement des outils
        self._setup_tools()
    
    def _setup_tools(self):
        """Déclare les fonctions disponibles pour les clients MCP"""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="validate_nmap_command",
                    description="Valide et corrige une commande Nmap (Sémantique + LLM)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string"},
                            "command": {"type": "string"},
                            "agent_name": {"type": "string"}
                        },
                        "required": ["intent", "command", "agent_name"]
                    }
                ),
                Tool(
                    name="execute_in_sandbox",
                    description="Exécute la commande dans un conteneur Docker isolé",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer"}
                        },
                        "required": ["command"]
                    }
                ),
                Tool(
                    name="execute_in_vm",
                    description="Exécute la commande finale sur la VM cible via SSH",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "target": {"type": "string"}
                        },
                        "required": ["command", "target"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Routeur des appels d'outils"""
            print(f"📡 MCP Tool Call: {name}")
            
            if name == "validate_nmap_command":
                # 1. Validation
                result = self.validator.validate(arguments["intent"], arguments["command"])
                
                # 2. Correction Automatique (Logique intégrée ici pour le tool)
                final_cmd = arguments["command"]
                corrected = False
                
                # Si recoverable + erreur root, on tente sudo
                if result.status == ValidationStatus.RECOVERABLE:
                    if any("root" in str(e).lower() for e in result.semantic_errors):
                        if "sudo" not in final_cmd:
                            final_cmd = "sudo " + final_cmd
                            corrected = True
                            # Re-validation rapide
                            result = self.validator.validate(arguments["intent"], final_cmd)

                response_data = {
                    "status": result.status.value,
                    "score": result.final_score,
                    "final_command": final_cmd,
                    "corrected": corrected,
                    "errors": result.semantic_errors,
                    "llm_reasoning": result.llm_reasoning
                }
                return [TextContent(type="text", text=json.dumps(response_data))]

            elif name == "execute_in_sandbox":
                res = await self.sandbox.execute(
                    arguments["command"], 
                    timeout=arguments.get("timeout", 60)
                )
                return [TextContent(type="text", text=json.dumps(res))]

            elif name == "execute_in_vm":
                # Utilisation du context manager de votre VMExecutor
                try:
                    with self.vm as vm:
                        res = vm.execute(arguments["command"], arguments["target"])
                    return [TextContent(type="text", text=json.dumps(res))]
                except Exception as e:
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]
            
            raise ValueError(f"Outil inconnu: {name}")

    async def run_sse(self, host="0.0.0.0", port=8000):
        """Lance le serveur Web SSE"""
        from mcp.server.sse import SseServerTransport
        
        async def handle_sse(request):
            transport = SseServerTransport("/messages")
            async with transport.connect_sse(request, self.server.create_initialization_options()) as streams:
                await self.server.run(streams[0], streams[1], self.server.create_initialization_options())
        
        # Note: L'implémentation exacte de SSE avec Starlette peut varier selon la version de mcp
        # Ceci est une structure compatible standard.
        app = Starlette(routes=[Route("/sse", endpoint=handle_sse)])
        
        import uvicorn
        print(f"🚀 MCP Server running on http://{host}:{port}/sse")
        config = uvicorn.Config(app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()