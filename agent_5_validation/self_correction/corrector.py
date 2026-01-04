"""
Step 9: Self-Correction Agent
Iterative correction loop with decoupled dependencies
"""
from typing import Tuple, List, Callable, Any

class SelfCorrectionAgent:
    """Autonomous self-correction loop"""
    
    def __init__(self, llm_generate_func: Callable, max_retries: int = 3):
        self.generate = llm_generate_func
        self.max_retries = max_retries
    
    async def correct(
        self,
        intent: str,
        failed_command: str,
        errors: List[str],
        mcp_client: Any  # Use 'Any' or a String 'MCPClient' to avoid importing the class
    ) -> Tuple[str, List[str]]:
        
        history = [f"Original: {failed_command}"]
        current_cmd = failed_command
        
        for attempt in range(self.max_retries):
            feedback = f"""
Previous command failed validation.
Errors: {errors}

Fix these errors while maintaining intent: "{intent}"
Return only the corrected command.
"""
            # Generate correction
            corrected = self.generate(intent, failed_command, feedback)
            history.append(f"Attempt {attempt + 1}: {corrected}")
            
            # Validate correction via the passed client
            # The client should have a validate_command method
            validation_data = await mcp_client.validate_command(
                command=corrected,
                intent=intent,
                agent_name="self-corrector"
            )
            
            # Note: handle both object-style and dict-style responses
            is_valid = validation_data.valid if hasattr(validation_data, 'valid') else validation_data.get('valid')
            
            if is_valid:
                return corrected, history
            
            current_cmd = corrected
            errors = validation_data.errors if hasattr(validation_data, 'errors') else validation_data.get('errors', [])
        
        history.append("⚠️ Max retries exceeded")
        return current_cmd, history