"""
Step 3: Hybrid Validator
Combines semantic + LLM (40% semantic, 60% LLM)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

from .llm_judge import MistralAPIValidator
from .semantic_validator import AdvancedSemanticValidator, SemanticValidationResult

class ValidationStatus(Enum):
    VALID = "valid"
    RECOVERABLE = "recoverable"
    INVALID = "invalid"

@dataclass
class HybridValidationResult:
    status: ValidationStatus
    semantic_score: float
    llm_score: float
    final_score: float
    semantic_errors: List[str]
    llm_reasoning: str
    suggestions: List[str]
    method_breakdown: Dict[str, float]  # {"semantic": 40, "llm": 60}
    confidence: float

class AdvancedHybridValidator:
    """
    Advanced Hybrid Validator - ALWAYS uses both Semantic + Mistral API fhemtini?
    
    Decision Logic:
    - Semantic score provides fast baseline (40% weight)
    - Mistral API provides contextual analysis (60% weight)
    - Combined scoring with confidence weighting
    """
    
    def __init__(self, mistral_api_url: str = "http://192.168.11.1:1234/v1/chat/completions"):
        self.semantic = AdvancedSemanticValidator()
        self.mistral = MistralAPIValidator(api_url=mistral_api_url)
        
        # Weights for hybrid scoring
        self.semantic_weight = 0.40  # 40% semantic
        self.llm_weight = 0.60       # 60% Mistral
    
    def validate(self, query: str, command: str) -> HybridValidationResult:
        """
        ALWAYS performs both validations and combines results
        
        Returns comprehensive hybrid validation result
        """
        print(f"\n🔍 Hybrid Validation Pipeline")
        print(f"{'='*60}")
        
        # STEP 1: Semantic Validation (Fast)
        print("  [1/2] Running semantic validation (rules + regex)...")
        semantic_result = self.semantic.validate(query, command)
        print(f"        → Semantic score: {semantic_result.score:.1f}/100")
        
        # STEP 2: Mistral API Validation (ALWAYS)
        print("  [2/2] Calling Mistral API for contextual analysis...")
        llm_result = self.mistral.validate(query, command, semantic_result)
        print(f"        → LLM confidence: {llm_result.confidence:.2f}")
        print(f"        → LLM verdict: {'✅ Valid' if llm_result.valid else '❌ Invalid'}")
        
        # STEP 3: Combine Scores
        # LLM score = confidence * 100 if valid, else confidence * semantic_score
        if llm_result.valid:
            llm_score = llm_result.confidence * 100
        else:
            # Penalize based on severity
            severity_penalty = {
                "info": 0.9,
                "warning": 0.7,
                "error": 0.4
            }
            llm_score = llm_result.confidence * semantic_result.score * severity_penalty.get(llm_result.severity, 0.5)
        
        # Weighted average
        final_score = (
            self.semantic_weight * semantic_result.score +
            self.llm_weight * llm_score
        )
        
        print(f"\n  📊 Score Breakdown:")
        print(f"     Semantic: {semantic_result.score:.1f} (weight: {self.semantic_weight})")
        print(f"     LLM:      {llm_score:.1f} (weight: {self.llm_weight})")
        print(f"     Final:    {final_score:.1f}/100")
        
        # STEP 4: Determine Status
        if final_score >= 80 and llm_result.valid:
            status = ValidationStatus.VALID
        elif final_score >= 38:
            status = ValidationStatus.RECOVERABLE
        else:
            status = ValidationStatus.INVALID
        
        # STEP 5: Collect All Errors and Suggestions
        all_errors = semantic_result.technical_errors.copy()
        if semantic_result.conflicts:
            all_errors.extend(semantic_result.conflicts)
        if semantic_result.missing_concepts:
            all_errors.extend([
                f"Missing concept: {m['concept']}" 
                for m in semantic_result.missing_concepts
            ])
        
        suggestions = []
        if llm_result.suggestion:
            suggestions.append(f"LLM: {llm_result.suggestion}")
        if not llm_result.valid:
            suggestions.append(f"Reasoning: {llm_result.reasoning}")
        
        return HybridValidationResult(
            status=status,
            semantic_score=semantic_result.score,
            llm_score=llm_score,
            final_score=final_score,
            semantic_errors=all_errors,
            llm_reasoning=llm_result.reasoning,
            suggestions=suggestions,
            method_breakdown={
                "semantic_contribution": self.semantic_weight * semantic_result.score,
                "llm_contribution": self.llm_weight * llm_score,
                "llm_confidence": llm_result.confidence
            },
            confidence=llm_result.confidence
        )
