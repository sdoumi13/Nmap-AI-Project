"""
MCP Client pour agents Easy/Medium/Hard
"""

# Fichier: mcp_tools/mcp_client.py
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

class MCPClient: 
    """
    Client MCP Officiel.
    Se connecte au serveur Agent 5 via SSE pour demander des validations.
    """
    
    def __init__(self, server_url: str = "http://localhost:8000/sse"):
        self.server_url = server_url
        self.session = None
        self.exit_stack = None

    async def connect(self):
        """Établit la connexion SSE persistante"""
        print(f"🔌 Connexion au serveur MCP: {self.server_url}")
        # Note: La gestion de contexte asynchrone est requise par mcp
        self.transport_ctx = sse_client(self.server_url)
        self.streams = await self.transport_ctx.__aenter__()
        self.session_ctx = ClientSession(self.streams[0], self.streams[1])
        self.session = await self.session_ctx.__aenter__()
        await self.session.initialize()
        print("✅ Connecté au serveur MCP")

    async def validate_command(self, intent: str, command: str, agent_name: str) -> dict:
        """Appelle l'outil validate_nmap_command"""
        if not self.session:
            await self.connect()
            
        result = await self.session.call_tool(
            "validate_nmap_command",
            arguments={
                "intent": intent,
                "command": command,
                "agent_name": agent_name
            }
        )
        # MCP retourne une liste de contenus, on prend le premier texte (JSON)
        return json.loads(result.content[0].text)

    async def execute_sandbox(self, command: str) -> dict:
        if not self.session:
            await self.connect()
        result = await self.session.call_tool(
            "execute_in_sandbox", 
            arguments={"command": command}
        )
        return json.loads(result.content[0].text)

    async def close(self):
        """Ferme proprement la session"""
        if self.session:
            await self.session_ctx.__aexit__(None, None, None)
            await self.transport_ctx.__aexit__(None, None, None)
            print("🔌 Déconnecté du serveur MCP")