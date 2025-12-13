"""
Step 3: Hybrid Validator
Combines semantic + LLM (80% semantic, 20% LLM)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

from validation.llm_judge import LLMJudge
from validation.semantic_validator import SemanticValidationResult, SemanticValidator

class ValidationStatus(Enum):
    VALID = "valid"
    RECOVERABLE = "recoverable"
    INVALID = "invalid"

@dataclass
class ValidationResult:
    status: ValidationStatus
    score: float
    errors: List[str]
    warnings: List[str]
    reasoning: str
    method_used: str

class HybridValidator:
    """Main validator: Semantic first, LLM fallback"""
    
    def __init__(self, use_llm: bool = True):
        self.semantic = SemanticValidator()
        self.llm = LLMJudge() if use_llm else None
    
    def validate(self, query: str, command: str) -> ValidationResult:
        """
        Main validation pipeline
        """
        # Stage 1: Semantic validation
        semantic_result = self.semantic.validate(query, command)
        
        # Decide if LLM needed
        needs_llm = self._needs_llm(query, semantic_result)
        
        if needs_llm and self.llm:
            # Stage 2: LLM validation
            llm_result = self.llm.validate(query, command, semantic_result)
            return self._build_llm_result(llm_result, semantic_result)
        else:
            # Use semantic only
            return self._build_semantic_result(semantic_result)
    
    def _needs_llm(self, query: str, result: SemanticValidationResult) -> bool:
        """Decide if LLM reasoning needed"""
        has_negation = any(w in query.lower() for w in ["don't", "not", "avoid"])
        has_conflicts = len(result.conflicts) > 0
        borderline_score = 50 < result.score < 80
        
        return has_negation or has_conflicts or borderline_score
    
    def _build_semantic_result(self, result: SemanticValidationResult) -> ValidationResult:
        """Build result from semantic validation"""
        errors = []
        errors.extend(result.technical_errors)
        errors.extend([f"Missing: {m['concept']}" for m in result.missing_concepts])
        errors.extend(result.conflicts)
        
        if result.score >= 80:
            status = ValidationStatus.VALID
        elif result.score >= 50:
            status = ValidationStatus.RECOVERABLE
        else:
            status = ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            score=result.score,
            errors=errors,
            warnings=[],
            reasoning=f"Semantic: {len(result.matched_concepts)} concepts matched",
            method_used="semantic"
        )
    
    def _build_llm_result(self, llm: Dict, semantic: SemanticValidationResult) -> ValidationResult:
        """Build result from LLM validation"""
        is_valid = llm.get("valid", False)
        
        status = ValidationStatus.VALID if is_valid else ValidationStatus.RECOVERABLE
        score = 90.0 if is_valid else 60.0
        
        return ValidationResult(
            status=status,
            score=score,
            errors=[] if is_valid else [llm.get("reasoning", "")],
            warnings=[llm.get("suggestion", "")],
            reasoning=f"LLM: {llm.get('reasoning', '')}",
            method_used="llm"
        )
