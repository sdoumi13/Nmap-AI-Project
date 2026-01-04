"""
NMAP Discrete Diffusion Language Model
========================================
Implements iterative denoising for complex Nmap command generation.
Based on Discrete Diffusion principles adapted for structured command synthesis.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import re
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


# ============================================================================
# STEP 1: Command Structure Definition
# ============================================================================

class NmapCommandStructure:
    """
    Defines valid Nmap command structure and categories.
    Critical for guiding diffusion process.
    """
    
    SCAN_TYPES = ['-sS', '-sT', '-sU', '-sN', '-sF', '-sX', '-sA', '-sW', '-sM']
    PORT_SPECS = ['-p-', '-p', '-F', '--top-ports']
    OS_VERSION = ['-O', '-sV', '-A', '--osscan-guess', '--version-intensity']
    SCRIPTS = ['--script']
    TIMING = ['-T0', '-T1', '-T2', '-T3', '-T4', '-T5']
    IP_VERSION = ['-4', '-6']
    OTHER = ['-Pn', '-n', '-R', '--traceroute', '-sn', '-v', '-vv', '-D']
    
    # Conflict rules (for validation during generation)
    CONFLICTS = [
        {'-sS', '-sT'},  # Can't do both SYN and TCP connect
        {'-sS', '-sU'},  # Actually allowed together but rarely used
    ]
    
    # Dependencies
    REQUIRES_ROOT = ['-sS', '-sU', '-O', '-sF', '-sN', '-sX']
    
    @classmethod
    def get_all_flags(cls) -> List[str]:
        """Get all known flags"""
        return (cls.SCAN_TYPES + cls.PORT_SPECS + cls.OS_VERSION + 
                cls.SCRIPTS + cls.TIMING + cls.IP_VERSION + cls.OTHER)
    
    @classmethod
    def categorize_flag(cls, flag: str) -> str:
        """Identify which category a flag belongs to"""
        if flag in cls.SCAN_TYPES:
            return 'scan_type'
        elif any(flag.startswith(p) for p in cls.PORT_SPECS):
            return 'ports'
        elif flag in cls.OS_VERSION:
            return 'os_version'
        elif flag.startswith('--script'):
            return 'scripts'
        elif flag in cls.TIMING:
            return 'timing'
        elif flag in cls.IP_VERSION:
            return 'ip_version'
        else:
            return 'other'


# ============================================================================
# STEP 2: Data Preparation with Noise Generation
# ============================================================================

class NmapNoiseGenerator:
    """
    Generates noisy/incomplete command variants for training.
    This is CRUCIAL for discrete diffusion training.
    
    KEY FIX: Semantic-aware noise generation that creates meaningful progressions
    """
    
    def __init__(self, noise_levels: int = 5):
        self.noise_levels = noise_levels
        self.structure = NmapCommandStructure()
    
    def parse_command(self, command: str) -> Dict:
        """Parse command into structured components by category"""
        parts = command.split()
        
        # Extract target (usually last element with IP/domain pattern)
        target = parts[-1] if parts and ('.' in parts[-1] or ':' in parts[-1]) else '<target>'
        flags = [p for p in parts[1:] if p != target]  # Skip 'nmap' and target
        
        # Categorize flags for semantic ordering
        categorized = {
            'scan_type': [],
            'ports': [],
            'os_version': [],
            'scripts': [],
            'timing': [],
            'ip_version': [],
            'other': []
        }
        
        for flag in flags:
            category = self.structure.categorize_flag(flag)
            categorized[category].append(flag)
        
        return {
            'flags': flags,
            'target': target,
            'categorized': categorized,
            'full_command': command
        }
    
    def generate_noisy_sequence(self, clean_command: str) -> List[str]:
        """
        Generate semantically-meaningful progressive sequence.
        
        Order: scan_type → ports → os_version → scripts → timing → other → target
        This creates meaningful intermediate commands.
        """
        parsed = self.parse_command(clean_command)
        target = parsed['target']
        categorized = parsed['categorized']
        
        if len(parsed['flags']) == 0:
            return ['nmap', clean_command]
        
        # Build sequence in semantic order
        sequence = []
        
        # Step 0: Just nmap
        sequence.append('nmap')
        
        # Step 1: Add scan type first (most fundamental)
        current_flags = []
        if categorized['scan_type']:
            current_flags.extend(categorized['scan_type'])
            sequence.append(f"nmap {' '.join(current_flags)} {target}")
        
        # Step 2: Add port specification
        if categorized['ports']:
            current_flags.extend(categorized['ports'])
            sequence.append(f"nmap {' '.join(current_flags)} {target}")
        
        # Step 3: Add OS/version detection
        if categorized['os_version']:
            current_flags.extend(categorized['os_version'])
            sequence.append(f"nmap {' '.join(current_flags)} {target}")
        
        # Step 4: Add scripts
        if categorized['scripts']:
            current_flags.extend(categorized['scripts'])
            sequence.append(f"nmap {' '.join(current_flags)} {target}")
        
        # Step 5: Add timing
        if categorized['timing']:
            current_flags.extend(categorized['timing'])
            sequence.append(f"nmap {' '.join(current_flags)} {target}")
        
        # Step 6: Add IP version and other flags
        if categorized['ip_version']:
            current_flags.extend(categorized['ip_version'])
            sequence.append(f"nmap {' '.join(current_flags)} {target}")
        
        if categorized['other']:
            current_flags.extend(categorized['other'])
            sequence.append(f"nmap {' '.join(current_flags)} {target}")
        
        # Final: Clean command (ensure exact match)
        if sequence[-1] != clean_command:
            sequence.append(clean_command)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sequence = []
        for cmd in sequence:
            if cmd not in seen:
                seen.add(cmd)
                unique_sequence.append(cmd)
        
        return unique_sequence
    
    def create_training_pairs(self, clean_command: str, nl_query: str) -> List[Dict]:
        """
        Create (noisy_t, clean_t-1) training pairs from command.
        
        Returns:
            List of dicts with 'nl', 'noisy', 'target' keys
        """
        sequence = self.generate_noisy_sequence(clean_command)
        
        pairs = []
        for i in range(len(sequence) - 1):
            pairs.append({
                'nl': nl_query,
                'noisy': sequence[i],
                'target': sequence[i + 1],
                'step': i,
                'total_steps': len(sequence) - 1
            })
        
        return pairs


# ============================================================================
# STEP 3: Dataset
# ============================================================================

class NmapDiffusionDataset(Dataset):
    """
    Dataset for discrete diffusion training.
    Each sample is a (noisy_command, target_command) pair conditioned on NL.
    """
    
    def __init__(self, json_path: str, tokenizer, augment: bool = True):
        self.tokenizer = tokenizer
        self.noise_generator = NmapNoiseGenerator(noise_levels=7)  # Increased from 5
        
        # Load raw data
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        print(f"Loaded {len(raw_data)} raw command pairs")
        
        # Generate training pairs with noise
        self.training_pairs = []
        for item in raw_data:
            pairs = self.noise_generator.create_training_pairs(
                clean_command=item['output'],
                nl_query=item['input']
            )
            self.training_pairs.extend(pairs)
        
        print(f"Generated {len(self.training_pairs)} training pairs from noisy sequences")
        
        # Optional: Data augmentation
        if augment:
            self._augment_data()
    
    def _augment_data(self):
        """Add paraphrased versions"""
        original_size = len(self.training_pairs)
        augmented = []
        
        for pair in self.training_pairs[:original_size // 3]:  # Augment 1/3
            # Simple paraphrasing
            nl = pair['nl']
            if 'Scan' in nl:
                nl_aug = nl.replace('Scan', 'Check')
            elif 'Run' in nl:
                nl_aug = nl.replace('Run', 'Execute')
            else:
                continue
            
            augmented.append({
                **pair,
                'nl': nl_aug,
                'augmented': True
            })
        
        self.training_pairs.extend(augmented)
        print(f"Augmented to {len(self.training_pairs)} pairs")
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        pair = self.training_pairs[idx]
        
        # Format input with clear separation and instruction
        input_text = f"refine: {pair['noisy']} | query: {pair['nl']}"
        target_text = pair['target']
        
        # Tokenize
        input_enc = self.tokenizer(
            input_text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_enc = self.tokenizer(
            target_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_enc['input_ids'].squeeze(0),
            'attention_mask': input_enc['attention_mask'].squeeze(0),
            'labels': target_enc['input_ids'].squeeze(0),
            'step': pair['step'],
            'total_steps': pair['total_steps']
        }


# ============================================================================
# STEP 4: Discrete Diffusion Model
# ============================================================================

class NmapDiscreteDiffusionLM(nn.Module):
    """
    T5-based discrete diffusion model for iterative command refinement.
    
    Key idea: Model learns to predict next less-noisy command given:
    - Natural language query
    - Current noisy command
    """
    
    def __init__(self, model_name: str = 't5-small', use_adapter: bool = True):
        super().__init__()
        
        # Load pre-trained T5
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Try T5Tokenizer first, fall back to AutoTokenizer if SentencePiece not available
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        except ImportError:
            print("⚠️  SentencePiece not found, using T5TokenizerFast instead")
            from transformers import T5TokenizerFast
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        
        # Optional: Add lightweight adapter layers for efficiency
        if use_adapter:
            self._add_adapters()
        
        self.structure = NmapCommandStructure()
    
    def _add_adapters(self):
        """Add adapter layers to T5 (optional, for parameter efficiency)"""
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 layers for fine-tuning
        for param in self.model.decoder.block[-2:].parameters():
            param.requires_grad = True
        
        for param in self.model.encoder.block[-2:].parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Standard T5 forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate_next_step(
        self, 
        nl_query: str, 
        current_command: str,
        max_length: int = 128
    ) -> str:
        """
        Generate next refined command given current state.
        
        This is the core denoising step.
        """
        # Format input with clear structure
        input_text = f"refine: {current_command} | query: {nl_query}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        # Use greedy decoding for deterministic output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,  # Greedy decoding
                do_sample=False,  # Deterministic
                early_stopping=True
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process: Clean up output
        generated = self._clean_output(generated.strip())
        
        return generated
    
    def _clean_output(self, command: str) -> str:
        """Clean up generated command to ensure valid syntax"""
        parts = command.split()
        
        if not parts or parts[0] != 'nmap':
            return 'nmap'
        
        cleaned_parts = ['nmap']
        valid_flags = self.structure.get_all_flags()
        
        # Invalid patterns to filter out
        invalid_patterns = ['IPv6', 'IPv4', 'target', 'port', 'ports', 'script', 'OS', 'version']
        
        for part in parts[1:]:
            # Skip invalid text patterns
            if part in invalid_patterns or part.endswith('.'):
                continue
            
            # Keep valid nmap flags
            if part.startswith('-'):
                # Check if it's a known flag or starts with known prefix
                if any(part.startswith(flag) for flag in valid_flags):
                    cleaned_parts.append(part)
                elif part in ['-p', '--script', '-T']:  # Partial flags
                    cleaned_parts.append(part)
            # Keep targets (IPs, domains, CIDR)
            elif '.' in part or ':' in part or '/' in part:
                # Validate it looks like IP/domain (not random like "62.25")
                if part.count('.') >= 2 or ':' in part or part == '<target>':
                    cleaned_parts.append(part)
            # Keep port numbers if they follow -p
            elif part.isdigit() and len(cleaned_parts) > 1:
                if cleaned_parts[-1] in ['-p', '--top-ports']:
                    cleaned_parts.append(part)
        
        return ' '.join(cleaned_parts)


# ============================================================================
# STEP 5: Iterative Diffusion Sampler
# ============================================================================

class DiscreteDiffusionSampler:
    """
    Implements the iterative denoising loop for generation.
    
    This is the HEART of your diffusion system.
    """
    
    def __init__(self, model: NmapDiscreteDiffusionLM, max_steps: int = 10):
        self.model = model
        self.max_steps = max_steps
        self.structure = NmapCommandStructure()
    
    def sample(
        self, 
        nl_query: str,
        initial_command: str = 'nmap',
        verbose: bool = False
    ) -> Dict:
        """
        Iteratively refine command from noisy to clean.
        
        Returns:
            Dict with 'final_command', 'steps', 'trajectory'
        """
        # Extract target from query if present
        target = self._extract_target_from_query(nl_query)
        
        # Derive allowed flags from intent to constrain decoding
        allowed = self._intent_allowed_flags(nl_query)

        current_command = initial_command
        trajectory = [current_command]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Diffusion Sampling: {nl_query}")
            if target:
                print(f"Detected target: {target}")
            print(f"{'='*60}")
            print(f"Step 0: {current_command}")
        
        for step in range(1, self.max_steps + 1):
            # Generate next refinement
            next_command = self.model.generate_next_step(nl_query, current_command)
            # Enforce intent constraints on flags
            next_command = self._enforce_allowed_flags(next_command, allowed)
            
            # Ensure target is included if detected
            if target and target not in next_command:
                parts = next_command.split()
                if parts[-1] != target:  # Avoid duplicate
                    next_command = next_command + ' ' + target
            
            if verbose:
                print(f"Step {step}: {next_command}")
            
            trajectory.append(next_command)
            
            # Stopping criteria: No change or minimal change
            if self._is_converged(current_command, next_command):
                if verbose:
                    print(f"\n✓ Converged at step {step}")
                break
            
            # Also stop if command is getting worse (removing flags)
            if step > 2 and len(next_command.split()) < len(current_command.split()):
                if verbose:
                    print(f"\n⚠ Stopping: command degrading")
                break
            
            current_command = next_command
        
        if verbose:
            print(f"{'='*60}\n")
        
        return {
            'final_command': current_command,
            'steps': len(trajectory) - 1,
            'trajectory': trajectory,
            'nl_query': nl_query
        }

    def _intent_allowed_flags(self, query: str) -> Dict[str, List[str]]:
        """Map natural language intent to an allowed set of flags.
        Returns dict with categories and specific flags permitted.
        """
        q = query.lower()
        allowed = {
            'scan_type': [],
            'ports': [],
            'os_version': [],
            'scripts': [],
            'timing': [],
            'ip_version': [],
            'other': []
        }
        
        # Service-to-port mapping
        service_ports = {
            'http': 80, 'web': 80,
            'https': 443, 'ssl': 443,
            'ssh': 22, 'sftp': 22,
            'ftp': 21,
            'telnet': 23,
            'smtp': 25,
            'dns': 53,
            'dhcp': 67,
            'pop3': 110, 'pop': 110,
            'imap': 143,
            'ldap': 389,
            'https': 443,
            'smtps': 465,
            'mysql': 3306,
            'rdp': 3389, 'remote desktop': 3389,
            'postgresql': 5432, 'postgres': 5432,
            'mongodb': 27017,
            'redis': 6379,
            'elasticsearch': 9200,
            'docker': 2375,
            'kubernetes': 6443,
            'vnc': 5900,
            'http-proxy': 8080,
            'https-proxy': 8443
        }
        
        # Scan type
        if 'syn' in q or 'stealth' in q:
            allowed['scan_type'].append('-sS')
        if 'tcp' in q or 'connect' in q:
            allowed['scan_type'].append('-sT')
        if 'udp' in q:
            allowed['scan_type'].append('-sU')
        
        # Ports: Extract service names and explicit port numbers
        extracted_ports = []
        
        # Check for service names
        for service, port in service_ports.items():
            if service in q:
                extracted_ports.append(port)
        
        # Extract explicit port numbers, but filter out IP octets
        # Look for patterns like "port 80", "ports 80,443", "80, 443" but NOT in IPs
        port_patterns = [
            r'port\s+(\d+)',  # "port 80"
            r'ports\s+(\d+)',  # "ports 80"
            r'(?:for|on)\s+(\d+)(?:\s*,\s*|\s+)',  # "for 80," or "on 443 "
        ]
        
        for pattern in port_patterns:
            matches = re.findall(pattern, q)
            for match in matches:
                port_num = int(match)
                if 1 <= port_num <= 65535 and port_num not in extracted_ports:
                    extracted_ports.append(port_num)
        
        # Also handle comma-separated lists like "80,443" or "80, 443"
        # but only if they appear before the IP/CIDR
        comma_pattern = r'(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?'
        comma_matches = re.search(comma_pattern, q)
        if comma_matches:
            for group in comma_matches.groups():
                if group:
                    port_num = int(group)
                    if 1 <= port_num <= 65535 and port_num not in extracted_ports:
                        extracted_ports.append(port_num)
        
        # Build port flag
        if extracted_ports:
            # Deduplicate and sort
            extracted_ports = sorted(list(set(extracted_ports)))
            port_str = ','.join(map(str, extracted_ports))
            allowed['ports'].append(f'-p {port_str}')
        
        # All ports
        if 'all ports' in q or '65535' in q:
            allowed['ports'].append('-p-')
        if 'fast scan' in q or 'quick scan' in q or 'rapid scan' in q:
            allowed['ports'].append('-F')
        # Top ports with number extraction
        if 'top' in q and 'ports' in q:
            # Try to extract the number from "top N ports"
            top_ports_match = re.search(r'top\s+(\d+)\s+ports', q, re.IGNORECASE)
            if top_ports_match:
                top_num = top_ports_match.group(1)
                allowed['ports'].append(f'--top-ports {top_num}')
            else:
                # Default to 100 if no number specified
                allowed['ports'].append('--top-ports 100')
        
        # OS / Version
        if 'os detection' in q or 'operating system' in q or 'os fingerprint' in q or ' os' in q or 'with os' in q or 'fingerprint' in q:
            allowed['os_version'].append('-O')
        if 'version detection' in q or 'service version' in q or 'service detection' in q:
            allowed['os_version'].append('-sV')
        # Broaden match: any mention of 'version' implies -sV
        if 'version' in q:
            if '-sV' not in allowed['os_version']:
                allowed['os_version'].append('-sV')
        if 'aggressive' in q and ('scan' in q or 'mode' in q):
            allowed['os_version'].append('-A')
        
        # Decoy and evasion
        if 'decoy' in q or 'bypass firewall' in q or 'bypass the firewall' in q or 'evade' in q or 'evasion' in q:
            allowed['other'].append('-D')
        
        # Scripts
        if 'script' in q or 'nse' in q:
            allowed['scripts'].append('--script')
        # Timing
        if 'timing' in q or 't5' in q or 'insane' in q or 'aggressive timing' in q:
            allowed['timing'].append('-T5')
        # IP version
        if 'ipv6' in q or 'v6' in q:
            allowed['ip_version'].append('-6')
        if 'ipv4' in q or 'v4' in q:
            allowed['ip_version'].append('-4')
        # Discovery
        if 'ping' in q or 'host discovery' in q or 'ping sweep' in q:
            allowed['other'].append('-sn')
        if 'traceroute' in q:
            allowed['other'].append('--traceroute')
        if 'no ping' in q or 'skip host discovery' in q:
            allowed['other'].append('-Pn')
        return allowed

    def _enforce_allowed_flags(self, command: str, allowed: Dict[str, List[str]]) -> str:
        """Filter generated command to include only flags allowed by intent.
        Also ensures script parameter presence if --script appears.
        Additionally adds missing intent-required flags not present in generation.
        Handles composite flags like '-p 80,443'.
        """
        parts = command.split()
        if not parts or parts[0] != 'nmap':
            return 'nmap'
        keep = ['nmap']
        target = None
        
        # Build set of allowed flags (handle composite flags like '-p 80,443')
        allowed_set = set(sum(allowed.values(), []))
        allowed_base = set()
        allowed_with_ports = {}  # Store port lists from allowed
        
        for flag in allowed_set:
            # Extract base flag (e.g., '-p' from '-p 80,443')
            if ' ' in flag:
                parts_flag = flag.split()
                base = parts_flag[0]
                allowed_base.add(base)
                allowed_with_ports[base] = parts_flag[1:]  # Store the port list
            else:
                allowed_base.add(flag)
        
        # Parse existing command and keep only allowed flags
        i = 1
        while i < len(parts):
            p = parts[i]
            if p.startswith('-'):
                # Check if base flag is allowed
                if p in allowed_base or p in allowed_set:
                    keep.append(p)
                    # If this flag expects a port list (like -p), grab following port numbers
                    if p in allowed_with_ports and i + 1 < len(parts):
                        # Peek ahead for port numbers
                        next_p = parts[i + 1]
                        if next_p and (next_p[0].isdigit() or next_p == '<target>'):
                            if next_p[0].isdigit():
                                keep.append(next_p)
                                i += 1
                # Move to next
                i += 1
                continue
            # numeric or target
            if p.isdigit():
                # Keep only if last flag expects number
                if keep[-1:] and keep[-1] in ['-p', '--top-ports']:
                    keep.append(p)
            elif '.' in p or ':' in p or '/' in p or p == '<target>':
                target = p
            i += 1
        
        # Ensure script parameter if script intent exists
        intent_scripts = '--script' in allowed_set
        if intent_scripts and '--script' not in keep:
            keep.append('--script')
        if '--script' in keep:
            # If no script name present, default to 'default'
            if 'default' not in parts and 'vuln' not in parts and 'discovery' not in parts and 'safe' not in parts and 'malware' not in parts and 'auth' not in parts:
                keep.extend(['--script', 'default'])
                # Remove duplicate --script if present twice
                seen = False
                keep_filtered = []
                for k in keep:
                    if k == '--script':
                        if seen:
                            continue
                        seen = True
                    keep_filtered.append(k)
                keep = keep_filtered
        
        # Timing: only include -T5 if explicitly allowed
        keep = [k for k in keep if not (k.startswith('-T') and k != '-T5' and '-T5' not in allowed_set)]
        
        # Add missing required flags from allowed intent (semantic order)
        def add_if_missing(flag):
            # Handle composite flags like '-p 80,443'
            if ' ' in flag:
                parts_flag = flag.split()
                base = parts_flag[0]
                # Check if this base flag is already present
                already_has_base = any(k == base for k in keep)
                if already_has_base:
                    # Replace bare flag with composite
                    keep_new = []
                    for k in keep:
                        if k == base:
                            keep_new.extend(parts_flag)
                        else:
                            keep_new.append(k)
                    keep[:] = keep_new
                else:
                    keep.extend(parts_flag)
            else:
                if flag not in keep:
                    keep.append(flag)
        
        # Semantic ordering: scan_type → ports → os_version → scripts → timing → ip_version → other
        for cat in ['scan_type','ports','os_version','scripts','timing','ip_version','other']:
            for flag in allowed.get(cat, []):
                add_if_missing(flag)
        
        # Handle -D flag: if present without decoy list, add default decoy IPs
        if '-D' in keep:
            d_index = keep.index('-D')
            # Check if next element is decoy IPs (contains commas or starts with digit)
            if d_index + 1 < len(keep):
                next_elem = keep[d_index + 1]
                # If next element is not already decoy IPs, it might be another flag or target
                if not (next_elem[0].isdigit() or ',' in next_elem):
                    # Insert default decoy
                    keep.insert(d_index + 1, '10.0.0.1,10.0.0.2')
            else:
                # No decoy specified, add default
                keep.append('10.0.0.1,10.0.0.2')
        
        # Deduplicate flags while preserving order (but keep sequences like '-p 80,443')
        dedup = []
        seen = set()
        skip_next = False
        for i, k in enumerate(keep):
            if skip_next:
                skip_next = False
                continue
            if k not in seen or k == 'nmap':
                dedup.append(k)
                seen.add(k)
                # If this is a flag expecting a value, keep the next item if it's a number/port list
                if k in ['-p', '--top-ports'] and i + 1 < len(keep):
                    next_k = keep[i + 1]
                    if next_k and (next_k[0].isdigit() or ',' in next_k):
                        dedup.append(next_k)
                        seen.add(next_k)
                # Handle -D decoy IPs
                if k == '-D' and i + 1 < len(keep):
                    next_k = keep[i + 1]
                    if next_k and (next_k[0].isdigit() or ',' in next_k):
                        dedup.append(next_k)
                        seen.add(next_k)
        keep = dedup
        
        # Reconstruct
        if target:
            if keep[-1] != target:
                keep.append(target)
        return ' '.join(keep)
    
    def _extract_target_from_query(self, query: str) -> str:
        """Extract IP/domain target from natural language query"""
        import re
        
        # IPv4 pattern with CIDR
        ipv4_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b'
        # IPv6 pattern (simplified)
        ipv6_pattern = r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
        # Domain pattern
        domain_pattern = r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'
        
        # Try IPv4
        match = re.search(ipv4_pattern, query)
        if match:
            return match.group(0)
        
        # Try IPv6  
        match = re.search(ipv6_pattern, query)
        if match:
            return match.group(0)
        
        # Try domain
        match = re.search(domain_pattern, query)
        if match and not any(kw in match.group(0).lower() for kw in ['scan', 'port', 'detect']):
            return match.group(0)
        
        # Default placeholder for queries without explicit targets
        return '<target>'
    
    def _is_converged(self, cmd1: str, cmd2: str, threshold: int = 3) -> bool:
        """Check if generation has converged"""
        # Exact match (strict)
        if cmd1.strip() == cmd2.strip():
            return True
        
        # Check if only minor differences (e.g., spacing)
        parts1 = cmd1.strip().split()
        parts2 = cmd2.strip().split()
        
        # If length difference is small and most tokens are the same
        if len(parts1) == len(parts2):
            differences = sum(1 for p1, p2 in zip(parts1, parts2) if p1 != p2)
            if differences <= 1:  # Allow 1 difference (e.g., target)
                return True
        
        return False
    
    def batch_sample(self, queries: List[str], verbose: bool = False) -> List[Dict]:
        """Sample multiple queries"""
        results = []
        for query in queries:
            result = self.sample(query, verbose=verbose)
            results.append(result)
        return results


# ============================================================================
# STEP 6: Training Loop
# ============================================================================

class DiffusionTrainer:
    """Handles training of the discrete diffusion model"""
    
    def __init__(
        self,
        model: NmapDiscreteDiffusionLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,  # Increased from 5e-5
            weight_decay=0.01
        )
    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int, save_path: str = 'nmap_diffusion_model'):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        self.model.model.save_pretrained(path)
        self.model.tokenizer.save_pretrained(path)


# ============================================================================
# STEP 7: Main Training Script
# ============================================================================

def main(json_path='nmap_commands.json', batch_size=8, num_epochs=20):
    """Complete training pipeline"""
    
    print("="*60)
    print("NMAP DISCRETE DIFFUSION MODEL - TRAINING")
    print("="*60)
    
    # Configuration
    JSON_PATH = json_path
    BATCH_SIZE = batch_size
    NUM_EPOCHS = num_epochs
    
    # Force NVIDIA GPU (cuda:0 is typically the discrete GPU)
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'  # Explicitly use GPU 0 (NVIDIA)
        torch.cuda.set_device(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n✓ Using GPU: {gpu_name}")
    else:
        DEVICE = 'cpu'
        print(f"\n⚠ No CUDA available, using CPU")
    
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}\n")
    
    # Initialize model
    print("Initializing model...")
    model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)  # Changed to False
    tokenizer = model.tokenizer
    
    # Prepare data
    print("\nPreparing dataset...")
    dataset = NmapDiffusionDataset(JSON_PATH, tokenizer, augment=True)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Train
    trainer = DiffusionTrainer(model, train_loader, val_loader, device=DEVICE)
    trainer.train(num_epochs=NUM_EPOCHS, save_path='nmap_diffusion_checkpoint')
    
    # Test sampling
    print("\n" + "="*60)
    print("TESTING GENERATION")
    print("="*60 + "\n")
    
    model.model.to(DEVICE)  # Ensure model is on GPU for inference
    sampler = DiscreteDiffusionSampler(model, max_steps=15)
    
    test_queries = [
        "Scan all ports with OS detection on 192.168.1.0/24",
        "Fast TCP scan on common ports with version detection",
        "Stealth SYN scan with script execution on IPv6 target"
    ]
    
    for query in test_queries:
        result = sampler.sample(query, verbose=True)
        print(f"Final: {result['final_command']}\n")


# ============================================================================
# STEP 8: Inference Script
# ============================================================================

def inference_demo():
    """Standalone inference script"""
    
    print("="*60)
    print("NMAP DISCRETE DIFFUSION - INFERENCE")
    print("="*60 + "\n")
    
    # Check if trained model checkpoint exists
    import os
    checkpoint_path = 'nmap_diffusion_checkpoint'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Model checkpoint not found at '{checkpoint_path}'")
        print(f"\nYou need to train the model first!")
        print(f"\nRun: python discrete_diffusion_nmap.py --mode train --epochs 20\n")
        return
    
    print(f"Loading trained model from {checkpoint_path}...")
    # Load trained model
    model = NmapDiscreteDiffusionLM(model_name=checkpoint_path, use_adapter=False)
    model.model.eval()
    print("Model loaded successfully!\n")
    
    sampler = DiscreteDiffusionSampler(model, max_steps=15)
    
    # Interactive mode
    print("Enter natural language queries (or 'quit' to exit):\n")
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        result = sampler.sample(query, verbose=True)
        
        print(f"\n{'='*60}")
        print(f"FINAL COMMAND: {result['final_command']}")
        print(f"Steps taken: {result['steps']}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NMAP Discrete Diffusion Model')
    parser.add_argument('--data', type=str, default='nmap_commands.json',
                       help='Path to JSON training data')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                       help='Run mode: train or inference')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Run training
        main(json_path=args.data, batch_size=args.batch_size, num_epochs=args.epochs)
    else:
        # Run inference
        inference_demo()