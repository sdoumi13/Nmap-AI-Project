"""
Self-Correction Agent
"""
from typing import Tuple, List, Callable, Any
import re


class SelfCorrectionAgent:
    """Simple, effective self-correction for Nmap commands"""
    
    def __init__(self, llm_generate_func: Callable = None, max_retries: int = 2):
        self.generate = llm_generate_func
        self.max_retries = max_retries
    
    async def correct(
        self,
        intent: str,
        failed_command: str,
        errors: List[str],
        mcp_client: Any = None
    ) -> Tuple[str, List[str]]:
        """
        Simple correction with rule-based fixes
        """
        
        history = [f"Original: {failed_command}"]
        current_cmd = failed_command
        
        print(f"\n🔧 Self-Correction Starting")
        print(f"   Intent: {intent}")
        print(f"   Errors: {errors}")
        
        for attempt in range(self.max_retries):
            print(f"\n  🔄 Attempt {attempt + 1}/{self.max_retries}")
            
            # Apply rule-based fixes
            corrected = self._apply_fixes(current_cmd, intent, errors)
            
            if corrected == current_cmd:
                print(f"     ℹ️  No automatic fix found")
                break
            
            print(f"     ✓ Applied fix: {corrected}")
            history.append(f"Attempt {attempt + 1}: {corrected}")
            
            # Validate the correction
            if mcp_client:
                try:
                    validation = await mcp_client.validate_command(
                        command=corrected,
                        intent=intent,
                        agent_name="self-corrector"
                    )
                    
                    is_valid = validation.get('valid', False)
                    score = validation.get('score', 0)
                    new_errors = validation.get('errors', [])
                    
                    print(f"     Score: {score}/100 | Valid: {is_valid}")
                    
                    if is_valid or score >= 75:
                        print(f"     ✅ Correction successful!")
                        return corrected, history
                    
                    # Update for next iteration
                    current_cmd = corrected
                    errors = new_errors
                    
                except Exception as e:
                    print(f"     ⚠️ Validation error: {e}")
                    return corrected, history
            else:
                # No validation available, return correction
                return corrected, history
        
        # If we get here, return the last attempt
        print(f"  ⚠️ Max retries reached")
        return corrected if corrected != failed_command else failed_command, history
    
    def _apply_fixes(self, command: str, intent: str, errors: List[str]) -> str:
        """
        Apply simple, rule-based fixes for common errors
        """
        
        corrected = command.strip()
        errors_str = ' '.join(str(e).lower() for e in errors)
        intent_lower = intent.lower()
        
        # FIX 1: Missing sudo for root-requiring flags
        if 'root privileges' in errors_str or 'permission' in errors_str:
            if not corrected.startswith('sudo'):
                print(f"     → Adding 'sudo' prefix")
                return f"sudo {corrected}"
        
        # FIX 2: Missing target
        if 'no target' in errors_str or 'target missing' in errors_str:
            target_match = re.search(r'\d+\.\d+\.\d+\.\d+', intent)
            if target_match and target_match.group(0) not in corrected:
                target = target_match.group(0)
                print(f"     → Adding target: {target}")
                return f"{corrected} {target}"
            elif 'TARGET' not in corrected.upper():
                print(f"     → Adding placeholder TARGET")
                return f"{corrected} TARGET"
        
        # FIX 3: Flag conflicts (e.g., -sS and -sT together)
        if '-sS' in corrected and '-sT' in corrected:
            print(f"     → Removing conflicting -sT flag")
            corrected = re.sub(r'\s+-sT\b', '', corrected)
            return corrected
        
        # FIX 4: Multiple timing flags
        timing_flags = ['-T0', '-T1', '-T2', '-T3', '-T4', '-T5']
        present = [t for t in timing_flags if t in corrected]
        if len(present) > 1:
            keep = present[-1]
            print(f"     → Keeping only {keep}, removing others")
            for t in present[:-1]:
                corrected = re.sub(rf'\s+{re.escape(t)}\b', '', corrected)
            return corrected
        
        # FIX 5: Missing required flags based on intent
        
        # OS detection
        if any(kw in intent_lower for kw in ['os', 'operating system', 'fingerprint']) and '-O' not in corrected:
            print(f"     → Adding -O flag for OS detection")
            corrected = re.sub(r'\bnmap\b', 'nmap -O', corrected, count=1)
            # Also add sudo if not present
            if not corrected.startswith('sudo'):
                corrected = f"sudo {corrected}"
            return corrected
        
        # Version detection
        if any(kw in intent_lower for kw in ['version', 'service']) and '-sV' not in corrected:
            print(f"     → Adding -sV flag for version detection")
            corrected = re.sub(r'\bnmap\b', 'nmap -sV', corrected, count=1)
            return corrected
        
        # UDP scan
        if 'udp' in intent_lower and '-sU' not in corrected:
            print(f"     → Adding -sU flag for UDP scan")
            corrected = re.sub(r'\bnmap\b', 'nmap -sU', corrected, count=1)
            if not corrected.startswith('sudo'):
                corrected = f"sudo {corrected}"
            return corrected
        
        # Stealth scan
        if any(kw in intent_lower for kw in ['stealth', 'stealthy', 'evade']) and not any(f in corrected for f in ['-sS', '-sF', '-sN']):
            print(f"     → Adding -sS flag for stealth scan")
            corrected = re.sub(r'\bnmap\b', 'nmap -sS', corrected, count=1)
            if not corrected.startswith('sudo'):
                corrected = f"sudo {corrected}"
            return corrected
        
        # Fragmentation (for evasion)
        if any(kw in intent_lower for kw in ['fragment', 'fragmentation', 'evade']) and '-f' not in corrected:
            print(f"     → Adding -f flag for packet fragmentation")
            corrected = re.sub(r'\bnmap\b', 'nmap -f', corrected, count=1)
            return corrected
        
        # FIX 6: Invalid decoy format
        if '-D' in corrected and any(kw in errors_str for kw in ['decoy', 'invalid']):
            print(f"     → Fixing decoy format to RND:10")
            corrected = re.sub(r'-D\s+\S+', '-D RND:10', corrected)
            return corrected
        
        # FIX 7: Invalid port syntax
        if 'port' in errors_str:
            # Fix common port errors like "80 443" -> "80,443"
            corrected = re.sub(r'-p\s+(\d+)\s+(\d+)', r'-p \1,\2', corrected)
            return corrected
        
        # No fix applied
        return command


# Test function
async def test_corrector():
    """Test the simplified corrector"""
    
    corrector = SelfCorrectionAgent()
    
    test_cases = [
        {
            "intent": "stealth scan with fragmentation",
            "command": "nmap -sS 192.168.1.1",
            "errors": ["Flags -sS require root privileges - missing 'sudo' prefix"]
        },
        {
            "intent": "detect OS on target",
            "command": "nmap 192.168.1.1",
            "errors": ["Missing required flags for OS detection"]
        },
        {
            "intent": "scan web ports",
            "command": "nmap -p 80 443 192.168.1.1",
            "errors": ["Invalid port syntax"]
        },
        {
            "intent": "stealth scan with decoys",
            "command": "nmap -sS -D 10.0.0.1,10.0.0.2 192.168.1.1",
            "errors": ["Flags -sS require root privileges"]
        }
    ]
    
    print("="*70)
    print("SELF-CORRECTION AGENT TEST")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Intent: {test['intent']}")
        print(f"Command: {test['command']}")
        print(f"Errors: {test['errors']}")
        
        corrected, history = await corrector.correct(
            intent=test['intent'],
            failed_command=test['command'],
            errors=test['errors'],
            mcp_client=None
        )
        
        print(f"\n✅ Final: {corrected}")
        print(f"History: {history}")
        print("-"*70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_corrector())