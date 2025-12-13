# Fichier: validation/llm_judge.py
import json
import requests
from typing import Dict

class LLMJudge:
    """LLM Judge using LM Studio local API"""
    
    def __init__(self, api_url: str = "http://192.168.11.1:1234/v1/chat/completions"):
        self.api_url = api_url
        self.model = "mistral-7b-instruct-v0.3"
    
    def validate(
        self, 
        query: str, 
        command: str, 
        semantic_result
    ) -> Dict:
        """
        Validate using LM Studio API (OpenAI compatible)
        """
        prompt = self._build_prompt(query, command, semantic_result)
        
        try:
            # LM Studio uses OpenAI-compatible API
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a Nmap security auditor. Respond only in JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Fallback if not valid JSON
                    return {
                        "valid": False,
                        "reasoning": content,
                        "severity": "error"
                    }
            else:
                print(f"LLM API Error: {response.status_code}")
                return {
                    "valid": False,
                    "reasoning": f"API error: {response.status_code}"
                }
        
        except Exception as e:
            print(f"LLM validation error: {e}")
            return {
                "valid": False,
                "reasoning": f"LLM error: {str(e)}"
            }
    
    def _build_prompt(self, query: str, command: str, semantic_result) -> str:
        """Build validation prompt"""
        matched = ", ".join(semantic_result.matched_concepts) if semantic_result.matched_concepts else "None"
        missing = "\n".join(
            f"- {m['concept']}: expects {m['expected_flags']}"
            for m in semantic_result.missing_concepts
        ) if semantic_result.missing_concepts else "None"
        
        return f"""Validate this Nmap command against user intent.

USER INTENT: "{query}"
GENERATED COMMAND: {command}

SEMANTIC ANALYSIS:
- Matched: {matched}
- Missing: {missing}
- Conflicts: {semantic_result.conflicts}
- Technical errors: {semantic_result.technical_errors}

Respond ONLY with this JSON structure:
{{
    "valid": true or false,
    "reasoning": "your detailed explanation",
    "severity": "info", "warning", or "error",
    "suggestion": "alternative command if invalid"
}}"""