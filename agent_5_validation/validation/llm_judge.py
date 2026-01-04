# File: agent_5_validation/validation/llm_judge.py
from dataclasses import dataclass
import json
import requests
from typing import Dict
from .semantic_validator import SemanticValidationResult

@dataclass
class LLMValidationResult:
    valid: bool
    reasoning: str
    severity: str
    suggestion: str
    confidence: float

class MistralAPIValidator:
    def __init__(self, api_url: str = "http://192.168.11.1:1234/v1/chat/completions"):
        self.api_url = api_url
        self.model = "mistral-7b-instruct-v0.3"
    
    def validate(self, query: str, command: str, semantic_result: SemanticValidationResult) -> LLMValidationResult:
        prompt = self._build_enhanced_prompt(query, command, semantic_result)
        
        # --- FIX: Merge System Prompt into User Prompt ---
        system_instruction = """You are an expert Nmap security auditor.
Respond ONLY in valid JSON format.
Your output must start with { and end with }."""

        full_content = f"{system_instruction}\n\nDATA TO ANALYZE:\n{prompt}"
        # -------------------------------------------------

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    # FIX: Only send one "user" message
                    "messages": [
                        {
                            "role": "user",
                            "content": full_content
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 600
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Cleanup markdown wrappers
                clean_content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    parsed = json.loads(clean_content)
                    return LLMValidationResult(
                        valid=parsed.get("valid", False),
                        reasoning=parsed.get("reasoning", "No reasoning provided"),
                        severity=parsed.get("severity", "warning"),
                        suggestion=parsed.get("suggestion", ""),
                        confidence=parsed.get("confidence", 0.7)
                    )
                except json.JSONDecodeError:
                    return LLMValidationResult(
                        valid=False,
                        reasoning=f"JSON Parse Error. Raw: {clean_content[:50]}...",
                        severity="error",
                        suggestion="",
                        confidence=0.5
                    )
            else:
                return LLMValidationResult(
                    valid=False,
                    reasoning=f"API error: {response.status_code} - {response.text}",
                    severity="error",
                    suggestion="",
                    confidence=0.0
                )
        
        except Exception as e:
            print(f"❌ Mistral API error: {e}")
            return LLMValidationResult(
                valid=False,
                reasoning=f"LLM validation failed: {str(e)}",
                severity="error",
                suggestion="",
                confidence=0.0
            )

    def _build_enhanced_prompt(self, query: str, command: str, semantic_result: SemanticValidationResult) -> str:
        # (Keep your existing _build_enhanced_prompt method exactly as it was)
        matched = ", ".join(semantic_result.matched_concepts) or "None"
        missing_str = "\n".join(f" - '{m['concept']}' flags: {m['expected_flags']}" for m in semantic_result.missing_concepts) or "None"
        conflicts_str = "\n".join(f" - {c}" for c in semantic_result.conflicts) or "None"
        errors_str = "\n".join(f" - {e}" for e in semantic_result.technical_errors) or "None"
        
        return f"""Validate this Nmap command comprehensively.

USER INTENT: "{query}"
GENERATED COMMAND: {command}

SEMANTIC ANALYSIS:
Matched: {matched}
Score: {semantic_result.score}/100
Missing:
{missing_str}
Conflicts:
{conflicts_str}
Technical Errors:
{errors_str}

TASK:
Analyze context, security, and intent.
RESPOND IN THIS JSON FORMAT:
{{
    "valid": true/false,
    "reasoning": "explanation",
    "severity": "info/warning/error",
    "suggestion": "alternative command",
    "confidence": 0.0-1.0
}}
"""