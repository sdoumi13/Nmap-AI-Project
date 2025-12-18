"""
Step 9: Self-Correction Agent
Iterative correction loop with LLM feedback
"""

from typing import Tuple, List, Callable

from mcp_tools.mcp_client import MCPClient

class SelfCorrectionAgent:
    """Autonomous self-correction loop"""
    
    def __init__(self, llm_generate_func: Callable, max_retries: int = 3):
        """
        Args:
            llm_generate_func: Function(intent, feedback) -> command
            max_retries: Max correction attempts
        """
        self.generate = llm_generate_func
        self.max_retries = max_retries
    
    async def correct(
        self,
        intent: str,
        failed_command: str,
        errors: List[str],
        mcp_client: MCPClient
    ) -> Tuple[str, List[str]]:
        """
        Iterative correction loop
        
        Returns: (corrected_command, history)
        """
        history = [f"Original: {failed_command}"]
        current_cmd = failed_command
        
        for attempt in range(self.max_retries):
            # Build feedback
            feedback = f"""
Previous command failed validation.
Errors: {errors}

Fix these errors while maintaining intent: "{intent}"
Return only the corrected command.
"""
            
            # Generate correction
            corrected = self.generate(intent, failed_command, feedback)
            history.append(f"Attempt {attempt + 1}: {corrected}")
            
            # Validate correction
            validation = await mcp_client.validate_command(
                command=corrected,
                intent=intent,
                agent_name="self-corrector"
            )
            
            if validation.valid:
                return corrected, history
            
            current_cmd = corrected
            errors = validation.errors
        
        history.append("⚠️ Max retries exceeded")
        return current_cmd, history

