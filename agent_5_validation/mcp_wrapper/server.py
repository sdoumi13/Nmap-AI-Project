# Step 5: MCP Server
# Fichier: agent_5_validation/mcp_wrapper/server.py
from dataclasses import asdict
from datetime import datetime
from .protocol import ValidationRequest, ValidationResponse
from validation.hybrid_validator import AdvancedHybridValidator, ValidationStatus

class MCPServer:
    """Centralized validation server"""
    
    def __init__(self):
        # Initialisation du validateur avancé avec l'URL Mistral
        self.validator = AdvancedHybridValidator(
            mistral_api_url="http://192.168.11.1:1234/v1/chat/completions"
        )
        self.message_log = []
    
    async def handle_validation(self, request: ValidationRequest) -> ValidationResponse:
        """Handle validation request"""
        start_time = datetime.now()
        
        # 1. Exécution de la validation
        result = self.validator.validate(request.intent, request.command)
        
        # 2. Construction de la réponse (MAPPING CORRECT ICI)
        # On fait correspondre les champs du 'AdvancedValidator' vers le 'Protocol'
        response = ValidationResponse(
            valid=(result.status == ValidationStatus.VALID),
            
            # CORRECTION 1 : 'final_score' au lieu de 'score'
            score=result.final_score, 
            status=result.status.value,
            errors=result.semantic_errors + ([f"LLM: {result.llm_reasoning}"] if not result.status == ValidationStatus.VALID else []),
            warnings=result.suggestions,
            
            method_used="hybrid (semantic + mistral)"
        )
        
        # 3. Logging
        self.message_log.append({
            "timestamp": start_time.isoformat(),
            "request": asdict(request),
            "response": asdict(response) 
        })
        
        return response