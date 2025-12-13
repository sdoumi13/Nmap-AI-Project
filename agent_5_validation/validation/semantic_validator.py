"""
Step 1: Semantic Validator
Fast, deterministic validation using concept-to-flag mapping
"""

import re
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SemanticValidationResult:
    matched_concepts: List[str]
    missing_concepts: List[Dict]
    conflicts: List[str]
    technical_errors: List[str]
    score: float

class SemanticValidator:
    """Rule-based validation using concept mapping"""
    
    def __init__(self):
        # Concept -> Flags mapping (Knowledge Graph lightweight)
        self.concept_map = {
            "stealth": ["-sS", "-sF", "-sN"],
            "tcp": ["-sT", "-sS"],
            "udp": ["-sU"],
            "version": ["-sV"],
            "os": ["-O"],
            "fast": ["-T4", "-T5"],
            "slow": ["-T0", "-T1", "-T2"],
            "web": ["-p 80,443", "-p 80-443"],
            "all ports": ["-p-", "-p 1-65535"],
        }
        
        # Flag rules (technical validation)
        self.flag_rules = {
            "-sS": {"requires": "root", "conflicts": ["-sT"]},
            "-sU": {"requires": "root", "conflicts": []},
            "-O": {"requires": "root", "conflicts": []},
        }
        
        # Semantic conflicts
        self.conflicts = {
            "stealth": ["fast", "aggressive"],
            "quiet": ["fast", "aggressive"],
        }
    
    def validate(self, query: str, command: str) -> SemanticValidationResult:
        """
        Main validation method
        Returns: SemanticValidationResult
        """
        # Extract concepts from query
        concepts = self._extract_concepts(query)
        
        # Check semantic match
        matched = []
        missing = []
        
        for concept in concepts:
            required_flags = self.concept_map.get(concept, [])
            if any(flag in command for flag in required_flags):
                matched.append(concept)
            else:
                missing.append({
                    "concept": concept,
                    "expected_flags": required_flags
                })
        
        # Check semantic conflicts
        conflicts = []
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                if c2 in self.conflicts.get(c1, []):
                    conflicts.append(f"'{c1}' conflicts with '{c2}'")
        
        # Check technical rules
        technical_errors = self._check_technical_rules(command)
        
        # Calculate score
        score = 100.0
        score -= len(missing) * 30
        score -= len(conflicts) * 20
        score -= len(technical_errors) * 10
        
        return SemanticValidationResult(
            matched_concepts=matched,
            missing_concepts=missing,
            conflicts=conflicts,
            technical_errors=technical_errors,
            score=max(0, score)
        )
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract concepts from natural language"""
        query_lower = query.lower()
        return [concept for concept in self.concept_map.keys() 
                if concept in query_lower]
    
    def _check_technical_rules(self, command: str) -> List[str]:
        """Check flag dependencies and conflicts"""
        errors = []
        flags = re.findall(r'-[A-Za-z0-9]+', command)
        
        for flag in flags:
            if flag in self.flag_rules:
                rule = self.flag_rules[flag]
                if rule["requires"] == "root":
                    errors.append(f"{flag} requires root privileges")
                
                for conflict in rule["conflicts"]:
                    if conflict in flags:
                        errors.append(f"{flag} conflicts with {conflict}")
        
        return errors
