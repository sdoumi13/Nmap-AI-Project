from enum import Enum
import re
from typing import Dict, List, Set
from dataclasses import dataclass


class ValidationStatus(Enum):
    VALID = "valid"
    RECOVERABLE = "recoverable"
    INVALID = "invalid"


@dataclass
class SemanticValidationResult:
    matched_concepts: List[str]
    missing_concepts: List[Dict]
    conflicts: List[str]
    technical_errors: List[str]
    score: float

@dataclass
class SemanticValidationResult:
    matched_concepts: List[str]
    missing_concepts: List[Dict]
    conflicts: List[str]
    technical_errors: List[str]
    score: float

class AdvancedSemanticValidator:
    """
    Validateur sémantique avancé avec Regex, IPv6, et analyse d'erreurs
    """
    
    def __init__(self):
        # Concept Map étendu
        self.concept_map = {
            # Scan Types
            "stealth": ["-sS", "-sF", "-sN", "-sX"],
            "stealthy": ["-sS", "-sF", "-sN"],
            "tcp": ["-sT", "-sS", "-sA"],
            "udp": ["-sU"],
            "syn": ["-sS"],
            "connect": ["-sT"],
            "fin": ["-sF"],
            "null": ["-sN"],
            "xmas": ["-sX"],
            
            # Detection
            "version": ["-sV", "-A"],
            "os": ["-O", "-A"],
            "os detection": ["-O", "-A"],
            "service": ["-sV"],
            "aggressive": ["-A"],
            
            # Scripts
            "vuln": ["--script vuln", "--script=vuln"],
            "vulnerability": ["--script vuln"],
            "script": ["--script"],
            "default scripts": ["-sC"],
            
            # Timing
            "fast": ["-T4", "-T5", "-F"],
            "quick": ["-T4", "-F"],
            "slow": ["-T0", "-T1", "-T2"],
            "polite": ["-T2"],
            "normal": ["-T3"],
            "insane": ["-T5"],
            "paranoid": ["-T0"],
            "sneaky": ["-T1"],
            
            # Ports
            "all ports": ["-p-", "-p 1-65535"],
            "common ports": ["-F", "--top-ports"],
            "web": ["-p 80,443", "-p 80-443"],
            "top ports": ["--top-ports"],
            
            # Protocol
            "ipv6": ["-6"],
            "ipv4": ["-4"],
            
            # Host Discovery
            "no ping": ["-Pn"],
            "ping scan": ["-sn"],
            "traceroute": ["--traceroute"],
            
            # Output
            "verbose": ["-v", "-vv"],
            "output": ["-oN", "-oX", "-oG", "-oA"],
        }
        
        # Règles techniques étendues
        self.flag_rules = {
            "-sS": {
                "requires": "root",
                "conflicts": ["-sT"],
                "description": "SYN Stealth Scan",
                "risk_level": "medium",
                "detection_risk": "low"
            },
            "-sU": {
                "requires": "root",
                "conflicts": [],
                "description": "UDP Scan",
                "risk_level": "low",
                "detection_risk": "low"
            },
            "-O": {
                "requires": "root",
                "conflicts": [],
                "description": "OS Detection",
                "risk_level": "high",
                "detection_risk": "high"
            },
            "-A": {
                "requires": "none",
                "conflicts": [],
                "description": "Aggressive Scan (OS+Version+Script+Traceroute)",
                "risk_level": "very_high",
                "detection_risk": "very_high"
            },
            "-T5": {
                "requires": "none",
                "conflicts": ["-T0", "-T1", "-T2", "-T3", "-T4"],
                "description": "Insane Timing",
                "risk_level": "high",
                "detection_risk": "very_high"
            },
            "-6": {
                "requires": "ipv6_support",
                "conflicts": ["-4"],
                "description": "IPv6 Scanning",
                "risk_level": "low",
                "detection_risk": "low"
            },
            "-Pn": {
                "requires": "none",
                "conflicts": ["-sn"],
                "description": "Skip host discovery",
                "risk_level": "low",
                "detection_risk": "low"
            },
        }
        
        # Semantic conflicts (intentions contradictoires)
        self.semantic_conflicts = {
            "stealth": ["fast", "aggressive", "insane"],
            "stealthy": ["fast", "aggressive", "insane"],
            "quiet": ["fast", "aggressive", "insane"],
            "slow": ["fast", "insane"],
            "ipv6": ["ipv4"],
        }
        
        # Port ranges validation regex
        self.port_pattern = re.compile(r'-p\s*(\d+(-\d+)?|\d+(,\d+)*|-)')
        
        # IPv6 address pattern (comprehensive)
        self.ipv6_pattern = re.compile(
            r'([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|'
            r'([0-9a-fA-F]{1,4}:){1,7}:|'
            r'([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|'
            r'::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}|'
            r'fe80::[0-9a-fA-F]{0,4}(?:%\w+)?'
        )
    
    def validate(self, query: str, command: str) -> SemanticValidationResult:
        """
        Validation sémantique complète avec regex avancés
        """
        concepts = self._extract_concepts(query)
        matched = []
        missing = []
        conflicts = []
        technical_errors = []
        
        # 1. Vérifier correspondance concepts -> flags
        for concept in concepts:
            possible_flags = self.concept_map.get(concept, [])
            found = False
            
            for flag in possible_flags:
                # Regex précis avec word boundaries
                escaped_flag = re.escape(flag)
                pattern = rf"(?:^|\s){escaped_flag}(?:\s|$|=)"
                
                if re.search(pattern, command):
                    found = True
                    break
            
            if found:
                matched.append(concept)
            else:
                missing.append({
                    "concept": concept,
                    "expected_flags": possible_flags
                })
        
        # 2. Détection IPv6 automatique
        has_ipv6_address = bool(self.ipv6_pattern.search(command))
        has_ipv6_flag = re.search(r'(?:^|\s)-6(?:\s|$)', command)
        
        if has_ipv6_address and not has_ipv6_flag:
            technical_errors.append("IPv6 address detected but '-6' flag missing")
            if not any(m['concept'] == 'ipv6' for m in missing):
                missing.append({"concept": "ipv6", "expected_flags": ["-6"]})
        
        # 3. Vérifier conflits sémantiques
        for i, concept1 in enumerate(concepts):
            conflicting = self.semantic_conflicts.get(concept1, [])
            for concept2 in concepts[i+1:]:
                if concept2 in conflicting:
                    conflicts.append(
                        f"Semantic conflict: '{concept1}' incompatible with '{concept2}'"
                    )
        
        # 4. Vérifier règles techniques
        tech_errors = self._check_technical_rules(command)
        technical_errors.extend(tech_errors)
        
        # 5. Valider syntaxe ports
        port_errors = self._validate_port_syntax(command)
        technical_errors.extend(port_errors)
        
        # 6. Calculer score sémantique
        score = 100.0
        score -= len(missing) * 20
        score -= len(conflicts) * 15
        score -= len(technical_errors) * 25
        
        return SemanticValidationResult(
            matched_concepts=matched,
            missing_concepts=missing,
            conflicts=conflicts,
            technical_errors=technical_errors,
            score=max(0.0, score)
        )
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract concepts with multi-word matching"""
        query_lower = query.lower()
        found = []
        
        # Sort by length (longer phrases first to avoid partial matches)
        sorted_concepts = sorted(self.concept_map.keys(), key=len, reverse=True)
        
        for concept in sorted_concepts:
            if concept in query_lower:
                found.append(concept)
        
        return found
    
    def _check_technical_rules(self, command: str) -> List[str]:
        """Check flag dependencies and conflicts"""
        errors = []
        present_flags = []
        
        for flag in self.flag_rules.keys():
            pattern = rf"(?:^|\s){re.escape(flag)}(?:\s|$)"
            if re.search(pattern, command):
                present_flags.append(flag)
        
        for flag in present_flags:
            rule = self.flag_rules[flag]
            
            # Check requirements
            if rule["requires"] == "root":
                if not command.strip().startswith("sudo"):
                    errors.append(
                        f"{flag} requires root privileges (add 'sudo' or use alternative)"
                    )
            
            # Check conflicts
            for conflict_flag in rule["conflicts"]:
                if conflict_flag in present_flags:
                    errors.append(
                        f"Flag conflict: {flag} cannot be used with {conflict_flag}"
                    )
        
        return errors
    
    def _validate_port_syntax(self, command: str) -> List[str]:
        """Validate port range syntax"""
        errors = []
        
        # Find -p flag with argument
        port_match = self.port_pattern.search(command)
        if port_match:
            port_spec = port_match.group(1)
            
            # Validate individual ports
            if ',' in port_spec:
                ports = port_spec.split(',')
                for port in ports:
                    if '-' in port:
                        # Range
                        try:
                            start, end = map(int, port.split('-'))
                            if start > end or start < 1 or end > 65535:
                                errors.append(f"Invalid port range: {port}")
                        except ValueError:
                            errors.append(f"Invalid port range syntax: {port}")
                    else:
                        # Single port
                        try:
                            p = int(port)
                            if p < 1 or p > 65535:
                                errors.append(f"Port out of range: {port}")
                        except ValueError:
                            errors.append(f"Invalid port number: {port}")
        
        return errors
    
    @staticmethod
    def analyze_stderr(stderr_output: str) -> str:
        """Analyze Nmap error output for specific issues"""
        if not stderr_output:
            return None
        
        stderr_lower = stderr_output.lower()
        
        error_patterns = {
            r"root privileges": "ROOT_REQUIRED",
            r"permission denied": "PERMISSION_DENIED",
            r"failed to resolve": "DNS_ERROR",
            r"network is unreachable": "NETWORK_DOWN",
            r"no route to host": "NO_ROUTE",
            r"host seems down": "HOST_DOWN",
            r"invalid argument": "SYNTAX_ERROR",
            r"unknown option": "UNKNOWN_FLAG",
        }
        
        for pattern, error_code in error_patterns.items():
            if re.search(pattern, stderr_lower):
                return error_code
        
        return "UNKNOWN_ERROR"
