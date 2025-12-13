# Step 5: MCP Server
from dataclasses import asdict
from datetime import datetime
from .protocol import ValidationRequest, ValidationResponse
from validation.hybrid_validator import HybridValidator, ValidationStatus


class MCPServer:
    """Centralized validation server"""
    
    def __init__(self):
        self.validator = HybridValidator(use_llm=True)
        self.message_log = []
    
    async def handle_validation(self, request: ValidationRequest) -> ValidationResponse:
        """Handle validation request"""
        start_time = datetime.now()
        
        # Validate
        result = self.validator.validate(request.intent, request.command)
        
        # Build response
        response = ValidationResponse(
            valid=(result.status == ValidationStatus.VALID),
            score=result.score,
            status=result.status.value,
            errors=result.errors,
            warnings=result.warnings,
            method_used=result.method_used
        )
        
        # Log
        self.message_log.append({
            "timestamp": start_time.isoformat(),
            "request": asdict(request),
            "response": asdict(response)
        })
        
        return response