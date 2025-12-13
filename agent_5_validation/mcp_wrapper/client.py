# Step 6: MCP Client (for agents)
from datetime import datetime
from .protocol import ValidationRequest, ValidationResponse
from .server import MCPServer


class MCPClient:
    """Client for agents to communicate with server"""
    
    def __init__(self, server: MCPServer):
        self.server = server
    
    async def validate_command(
        self, 
        command: str, 
        intent: str, 
        agent_name: str
    ) -> ValidationResponse:
        """Send validation request"""
        request = ValidationRequest(
            command=command,
            intent=intent,
            context={"agent": agent_name, "timestamp": datetime.now().isoformat()}
        )
        
        return await self.server.handle_validation(request)