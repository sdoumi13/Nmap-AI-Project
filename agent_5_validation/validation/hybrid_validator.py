"""
Advanced Hybrid Validator
Qwen2.5-Coder-3B for Self-Correction (Port 1234)
"""

import httpx
import asyncio
import re
import json
from typing import Dict, Any, List
from enum import Enum

# Colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class ValidationStatus(str, Enum):
    VALID = "valid"
    RECOVERABLE = "recoverable"
    INVALID = "invalid"


class SemanticValidator:
    """Rule-based semantic validation"""
    
    REQUIRED_PATTERNS = [
        r'nmap',  # Must contain nmap
        r'-[a-zA-Z]+',  # Must have at least one flag
    ]
    
    DANGEROUS_PATTERNS = [
        r'rm\s+-rf',
        r'sudo\s+rm',
        r'dd\s+if=',
        r'mkfs\.',
        r':(){ :|:& };:',  # Fork bomb
    ]
    
    # FIX: Add flags that require root
    ROOT_FLAGS = ['-sS', '-sU', '-O', '-A', '--osscan-limit', '--osscan-guess']
    
    NMAP_FLAGS = {
        '-sS': 'SYN scan',
        '-sT': 'TCP connect scan',
        '-sU': 'UDP scan',
        '-sV': 'Version detection',
        '-O': 'OS detection',
        '-p': 'Port specification',
        '-A': 'Aggressive scan',
        '--script': 'NSE script',
        '-f': 'Fragmentation',
        '-D': 'Decoy',
        '-T': 'Timing',
    }
    
    def validate(self, command: str) -> Dict[str, Any]:
        """Validate command using semantic rules"""
        
        errors = []
        warnings = []
        score = 100
        
        # Check dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                errors.append(f"Dangerous command detected: {pattern}")
                score = 0
        
        # Check for nmap
        if not re.search(r'\bnmap\b', command, re.IGNORECASE):
            errors.append("Command does not contain 'nmap'")
            score -= 50
        
        # Check for at least one flag
        if not re.search(r'-[a-zA-Z]+', command):
            warnings.append("No flags detected, may be incomplete")
            score -= 20
        
        # FIX: Check if root-requiring flags are present without sudo
        has_sudo = command.strip().startswith("sudo")
        root_flags_found = []
        
        for flag in self.ROOT_FLAGS:
            if re.search(rf'(?:^|\s){re.escape(flag)}(?:\s|$)', command):
                root_flags_found.append(flag)
        
        if root_flags_found and not has_sudo:
            errors.append(f"Flags {', '.join(root_flags_found)} require root privileges - missing 'sudo' prefix")
            score -= 30  # Major penalty
        
        # Check for target
        ip_pattern = r'\d+\.\d+\.\d+\.\d+(/\d+)?'
        domain_pattern = r'\b[a-z0-9.-]+\.[a-z]{2,}\b'
        
        has_ip = re.search(ip_pattern, command)
        has_domain = re.search(domain_pattern, command, re.IGNORECASE)
        
        if not has_ip and not has_domain and 'TARGET' not in command.upper():
            warnings.append("No target detected (IP/domain)")
            score -= 10
        
        # Determine status
        if score >= 80:
            status = ValidationStatus.VALID
        elif score >= 50:
            status = ValidationStatus.RECOVERABLE
        else:
            status = ValidationStatus.INVALID
        
        return {
            'status': status,
            'score': max(0, score),
            'errors': errors,
            'warnings': warnings,
            'method': 'semantic'
        }


class QwenJudge:
    """Qwen2.5-Coder-3B as LLM Judge for self-correction"""
    
    def __init__(self, api_url: str = "http://192.168.11.1:1234/v1/chat/completions"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.model_name = "qwen2.5-coder-3b-instruct"
    
    async def validate(self, command: str, intent: str) -> Dict[str, Any]:
        """Validate command using Qwen"""
        
        prompt = f"""You are an expert in Nmap network security tools.

**Task:** Validate if the Nmap command correctly implements the user's intent.

**User Intent:** {intent}

**Nmap Command:** {command}

**Validation Criteria:**
1. Command syntax is correct
2. Command matches the intent
3. No dangerous/destructive operations
4. Target is specified (or placeholder like TARGET)
5. Flags are appropriate for the task
6. **CRITICAL: Flags like -O, -sS, -sU, -A require root privileges (sudo prefix)**

**Response (JSON only, no markdown):**
{{
    "is_valid": true,
    "score": 85,
    "errors": [],
    "warnings": ["Consider adding -Pn if firewall blocks ping"],
    "reasoning": "Command correctly implements a version detection scan"
}}

Respond:"""

        try:
            response = await self.client.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a cybersecurity expert. Always respond in valid JSON without markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300
                }
            )
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            
            # Clean JSON
            content = content.strip()
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\n?', '', content)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(content)
            
            is_valid = parsed.get('is_valid', False)
            score = parsed.get('score', 50)
            
            # Map to ValidationStatus
            if score >= 80:
                status = ValidationStatus.VALID
            elif score >= 50:
                status = ValidationStatus.RECOVERABLE
            else:
                status = ValidationStatus.INVALID
            
            return {
                'status': status,
                'score': score,
                'errors': parsed.get('errors', []),
                'warnings': parsed.get('warnings', []),
                'reasoning': parsed.get('reasoning', 'Qwen validation'),
                'method': 'qwen_judge'
            }
        
        except httpx.ConnectError:
            print(f"{RED}[Qwen Judge] Connection failed{RESET}")
            return {
                'status': ValidationStatus.RECOVERABLE,
                'score': 50,
                'errors': ['Qwen validation unavailable'],
                'warnings': [],
                'reasoning': 'LLM offline - using fallback',
                'method': 'qwen_judge_fallback'
            }
        except Exception as e:
            print(f"{RED}[Qwen Judge] Error: {str(e)[:100]}{RESET}")
            return {
                'status': ValidationStatus.RECOVERABLE,
                'score': 50,
                'errors': [f'Qwen error: {str(e)[:50]}'],
                'warnings': [],
                'reasoning': 'Validation error',
                'method': 'qwen_judge_error'
            }
    
    async def suggest_correction(self, command: str, intent: str, errors: List[str]) -> str:
        """Suggest correction for invalid command"""
        
        prompt = f"""You are an expert in Nmap. Fix the Nmap command based on errors.

**User Intent:** {intent}

**Failed Command:** {command}

**Errors:**
{chr(10).join([f'- {e}' for e in errors])}

**Common fixes for root privilege errors:**
- Add 'sudo ' prefix before nmap
- Example: 'nmap -O target' → 'sudo nmap -O target'

**Task:** Provide a CORRECTED Nmap command that:
1. Fixes all errors
2. Matches the user intent
3. Uses proper Nmap syntax
4. **Adds 'sudo' if using -O, -sS, -sU, or -A flags**

**Response (JSON only):**
{{
    "corrected_command": "sudo nmap -O -sV TARGET",
    "changes_made": "Added sudo prefix for root privileges required by -O flag"
}}

Respond:"""

        try:
            response = await self.client.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a cybersecurity expert. Always respond in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 200
                }
            )
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            
            # Clean JSON
            content = content.strip()
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\n?', '', content)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(content)
            
            corrected = parsed.get('corrected_command', command)
            print(f"{GREEN}[Qwen] Suggested correction: {corrected}{RESET}")
            
            return corrected
        
        except Exception as e:
            print(f"{RED}[Qwen] Correction failed: {str(e)[:100]}{RESET}")
            # Fallback: add sudo if missing
            if not command.strip().startswith('sudo'):
                return f"sudo {command}"
            return command
    
    async def close(self):
        await self.client.aclose()


class AdvancedHybridValidator:
    """Hybrid validator combining semantic rules + Qwen"""
    
    def __init__(self, mistral_api_url: str = "http://192.168.11.1:1234/v1/chat/completions"):
        self.semantic_validator = SemanticValidator()
        self.qwen_judge = QwenJudge(api_url=mistral_api_url)
        print(f"{GREEN}[Hybrid Validator] Initialized with Qwen on {mistral_api_url}{RESET}")
    
    async def validate(self, command: str, intent: str, agent_name: str = "unknown") -> Dict[str, Any]:
        """Hybrid validation: Semantic + Qwen"""
        
        print(f"\n{CYAN}[Validation] Validating command from {agent_name}{RESET}")
        print(f"  Command: {command}")
        print(f"  Intent: {intent}")
        
        # Step 1: Semantic validation (fast)
        semantic_result = self.semantic_validator.validate(command)
        
        print(f"  [Semantic] Score: {semantic_result['score']}/100")
        if semantic_result['errors']:
            print(f"  [Semantic] Errors: {semantic_result['errors']}")
        
        # FIX: If semantic finds critical errors (root privileges), mark as RECOVERABLE
        has_root_error = any('root privileges' in str(e).lower() for e in semantic_result['errors'])
        
        if has_root_error:
            # Force RECOVERABLE status for root privilege issues
            print(f"  {YELLOW}[Semantic] Root privilege issue detected - marking RECOVERABLE{RESET}")
            return {
                'valid': False,
                'status': ValidationStatus.RECOVERABLE,
                'score': semantic_result['score'],
                'errors': semantic_result['errors'],
                'warnings': semantic_result['warnings'],
                'method_used': 'semantic'
            }
        
        # If semantic validation passes, we're good
        if semantic_result['status'] == ValidationStatus.VALID:
            print(f"  {GREEN}[Result] VALID (semantic rules passed){RESET}")
            return {
                'valid': True,
                'status': ValidationStatus.VALID,
                'score': semantic_result['score'],
                'errors': semantic_result['errors'],
                'warnings': semantic_result['warnings'],
                'method_used': 'semantic'
            }
        
        # If semantic failed badly, don't bother with LLM
        if semantic_result['status'] == ValidationStatus.INVALID:
            print(f"  {RED}[Result] INVALID (semantic rules failed){RESET}")
            return {
                'valid': False,
                'status': ValidationStatus.INVALID,
                'score': semantic_result['score'],
                'errors': semantic_result['errors'],
                'warnings': semantic_result['warnings'],
                'method_used': 'semantic'
            }
        
        # Step 2: Qwen Judge (accurate but slower)
        qwen_result = await self.qwen_judge.validate(command, intent)
        
        print(f"  [Qwen] Score: {qwen_result['score']}/100")
        if qwen_result['errors']:
            print(f"  [Qwen] Errors: {qwen_result['errors']}")
        
        # Combine scores (weighted average)
        final_score = (semantic_result['score'] * 0.4) + (qwen_result['score'] * 0.6)
        
        # Determine final status
        if final_score >= 80:
            final_status = ValidationStatus.VALID
        elif final_score >= 50:
            final_status = ValidationStatus.RECOVERABLE
        else:
            final_status = ValidationStatus.INVALID
        
        is_valid = (final_status == ValidationStatus.VALID)
        
        color = GREEN if is_valid else YELLOW if final_status == ValidationStatus.RECOVERABLE else RED
        print(f"  {color}[Result] {final_status.value.upper()} (score: {final_score:.0f}/100){RESET}")
        
        return {
            'valid': is_valid,
            'status': final_status,
            'score': int(final_score),
            'errors': semantic_result['errors'] + qwen_result['errors'],
            'warnings': semantic_result['warnings'] + qwen_result['warnings'],
            'method_used': 'hybrid',
            'reasoning': qwen_result.get('reasoning', '')
        }
    
    async def suggest_correction(self, command: str, intent: str, errors: List[str]) -> str:
        """Suggest correction using Qwen"""
        return await self.qwen_judge.suggest_correction(command, intent, errors)
    
    async def close(self):
        await self.qwen_judge.close()


# Test function
async def test_validator():
    validator = AdvancedHybridValidator()
    
    test_cases = [
        {
            "command": "nmap -sV -p 80,443 192.168.1.1",
            "intent": "detect service versions on web ports"
        },
        {
            "command": "nmap -O 192.168.1.1",  # Missing sudo
            "intent": "detect operating system"
        },
        {
            "command": "nmap 192.168.1.1",
            "intent": "scan target"
        },
        {
            "command": "rm -rf /",
            "intent": "scan target"
        },
        {
            "command": "nmap -sS -f -D RND:10 TARGET",  # Missing sudo
            "intent": "stealth scan with fragmentation and decoys"
        }
    ]
    
    print(f"\n{'='*70}")
    print("HYBRID VALIDATOR TEST")
    print(f"{'='*70}\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        result = await validator.validate(
            command=test['command'],
            intent=test['intent'],
            agent_name="test"
        )
        
        if not result['valid'] and result['status'] == ValidationStatus.RECOVERABLE:
            print(f"\n  {YELLOW}[Attempting Correction]{RESET}")
            corrected = await validator.suggest_correction(
                command=test['command'],
                intent=test['intent'],
                errors=result['errors']
            )
            print(f"  Corrected: {corrected}")
        
        print("-" * 70)
    
    await validator.close()

if __name__ == "__main__":
    asyncio.run(test_validator())