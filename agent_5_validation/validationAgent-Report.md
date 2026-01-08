# Part 5: Validation & Self-Correction Architecture

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Validation Pipeline](#validation-pipeline)
- [Self-Correction Mechanism](#self-correction-mechanism)
- [Execution Flow](#execution-flow)
- [Algorithms & Methods](#algorithms--methods)

---

## 🎯 Overview

Agent 5 implements a **4-stage pipeline** for safe and intelligent Nmap command execution:

1. **Validation** - Hybrid validation using semantic rules + LLM judge
2. **Self-Correction** - Automatic command fixing for recoverable errors
3. **Sandbox Testing** - Docker-based safe execution environment
4. **VM Execution** - Real Nmap execution on Ubuntu VM

![Validation & Self-Correction Flow](/Annexe//validation+self-correction.png)

---

## 🏗️ System Architecture

```
agent5_validation+self-correction-sandboxing-testVM/
├── __init__.py
├── validation/
│   ├── __init__.py
│   ├── semantic_validator.py      # Step 1: Rule-based validation
│   ├── llm_judge.py                # Step 2: LLM-based validation
│   └── hybrid_validator.py         # Step 3: Combined validation
├── mcp_tools/
│   ├── __init__.py
│   └── mcp_server.py              # Step 4-5-6: MCP server interface
├── execution/
│   ├── __init__.py
│   ├── sandbox_executor.py         # Step 7: Docker sandbox
│   └── vm_executor.py              # Step 8: VM SSH execution
├── self_correction/
│   ├── __init__.py
│   └── corrector.py                # Step 9: Self-correction agent
├── run_agent5.py                   # Step 10: Main orchestrator
└── agent5_config.yaml              # Configuration file
```

### Running the System

```bash
# Start Agent 5 MCP Server (Port 5002)
python run_agent5.py

# Or start MCP server independently
python mcp_tools/mcp_server.py
```

---

## 🔍 Validation Pipeline

### 1. Semantic Validator (Rule-Based)

**Purpose**: Fast, deterministic validation using regex patterns and semantic rules.

**Algorithm**:
```python
class SemanticValidator:
    def validate(command: str) -> ValidationResult:
        score = 100
        
        # Check 1: Dangerous patterns
        if matches_dangerous_pattern(command):
            return INVALID (score = 0)
        
        # Check 2: Root privilege flags
        if has_root_flags(command) and not has_sudo(command):
            score -= 30
            return RECOVERABLE
        
        # Check 3: Target detection
        if not has_target(command):
            score -= 10
        
        # Check 4: Flag validation
        if has_flag_conflicts(command):
            score -= 20
        
        # Final decision
        if score >= 80: return VALID
        elif score >= 50: return RECOVERABLE
        else: return INVALID
```

**Key Features**:
- ✅ Detects dangerous commands (rm -rf, fork bombs)
- ✅ Validates root privilege requirements
- ✅ Checks target presence (IP/domain)
- ✅ Identifies flag conflicts (-sS vs -sT)
- ✅ Validates port syntax

**Example**:
```bash
Input:  nmap -sS 192.168.1.1
Error:  "Flags -sS require root privileges - missing 'sudo' prefix"
Status: RECOVERABLE (score: 70/100)
```

---

### 2. LLM Judge (Qwen2.5-Coder-3B / Mistral-7B)
![LLM](/Annexe//LLM.png)

**Purpose**: Intelligent validation using language models for complex cases.

**Algorithm**:
```python
class QwenJudge:
    async def validate(command: str, intent: str) -> ValidationResult:
        prompt = f"""
        Validate if this Nmap command matches the user intent:
        
        Intent: {intent}
        Command: {command}
        
        Criteria:
        1. Syntax correctness
        2. Intent alignment
        3. Security concerns
        4. Root privilege requirements
        
        Response (JSON):
        {{
            "is_valid": bool,
            "score": int,
            "errors": [str],
            "warnings": [str],
            "reasoning": str
        }}
        """
        
        response = await llm_api_call(prompt)
        return parse_llm_response(response)
```

**Models Used**:
- **Qwen2.5-Coder-3B** (Port 1234) - Primary LLM judge
- **Mistral-7B** (Fallback) - Alternative validation

**Features**:
-  Context-aware validation
-  Intent matching verification
-  Confidence scoring
- 💡 Detailed reasoning

---

### 3. Hybrid Validator (Combined Approach)

**Purpose**: Best of both worlds - fast semantic rules + intelligent LLM validation.

**Algorithm**:
```python
class HybridValidator:
    async def validate(command: str, intent: str) -> ValidationResult:
        # Stage 1: Semantic validation (fast)
        semantic_result = semantic_validator.validate(command)
        
        # Decision tree
        if semantic_result.status == VALID:
            return semantic_result  # ✅ Fast path
        
        if semantic_result.status == INVALID:
            return semantic_result  # ❌ Fail fast
        
        if has_root_privilege_error(semantic_result):
            return RECOVERABLE  # 🔧 Can be fixed
        
        # Stage 2: LLM validation (accurate)
        llm_result = await qwen_judge.validate(command, intent)
        
        # Combine scores (weighted average)
        final_score = (semantic_result.score × 0.4) + 
                      (llm_result.score × 0.6)
        
        # Final decision
        if final_score >= 80: return VALID
        elif final_score >= 50: return RECOVERABLE
        else: return INVALID
```

**Decision Matrix**:
| Semantic Score | LLM Score | Final Status |
|----------------|-----------|--------------|
| 100            | -         | VALID      |
| 70 (root err)  | -         | RECOVERABLE 🔧 |
| < 50           | -         | INVALID ❌   |
| 60-79          | 80-100    | VALID      |
| 60-79          | 50-79     | RECOVERABLE 🔧 |

![Validation](/Annexe//Validation.png)

---

## 🔧 Self-Correction Mechanism

### Corrector Algorithm

**Purpose**: Automatically fix recoverable command errors.

**Flow Chart**:
```
┌─────────────────────┐
│  Invalid Command    │
│  (RECOVERABLE)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Analyze Error Type  │
└──────────┬──────────┘
           │
           ├─► Root Privilege Error    → Add 'sudo'
           ├─► Missing Target          → Add IP/TARGET
           ├─► Flag Conflict           → Remove conflicting flag
           ├─► Invalid Port Syntax     → Fix comma separation
           ├─► Missing Required Flag   → Add based on intent
           ├─► Invalid Decoy Format    → Fix to RND:10
           └─► Multiple Timing Flags   → Keep only one
           
           ▼
┌─────────────────────┐
│  Apply Fix          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Re-Validate        │
└──────────┬──────────┘
           │
           ├─► VALID (score ≥ 75)     → ✅ Success
           ├─► Still Invalid          → 🔄 Retry (max 2)
           └─► Max Retries            → ⚠️ Return best attempt
```

### Correction Strategies

#### 1. Root Privilege Fix
```python
Error:  "Flags -sS require root privileges"
Input:  nmap -sS 192.168.1.1
Output: sudo nmap -sS 192.168.1.1
```

#### 2. Missing Target Fix
```python
Error:  "No target detected"
Input:  nmap -sS -p 80,443
Output: nmap -sS -p 80,443 TARGET
```

#### 3. Flag Conflict Resolution
```python
Error:  "Flag conflict: -sS and -sT"
Input:  nmap -sS -sT 192.168.1.1
Output: nmap -sS 192.168.1.1  # Remove -sT
```

#### 4. Intent-Based Flag Addition
```python
Intent: "stealth scan with fragmentation"
Input:  nmap -sS 192.168.1.1
Output: sudo nmap -f -sS 192.168.1.1  # Add -f for fragmentation
```

#### 5. Port Syntax Correction
```python
Error:  "Invalid port syntax"
Input:  nmap -p 80 443 192.168.1.1
Output: nmap -p 80,443 192.168.1.1
```

#### 6. Timing Flag Deduplication
```python
Error:  "Multiple timing flags"
Input:  nmap -T3 -T4 -T5 192.168.1.1
Output: nmap -T5 192.168.1.1  # Keep last one
```

---

##  Execution Flow

### Complete Pipeline Example

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT                                   │
│  Intent: "stealth scan with fragmentation"                      │
│  Command: nmap -sS -D 10.0.0.1,10.0.0.2 192.168.188.128         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: VALIDATION VIA MCP                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Semantic Validator (Fast)                               │   │
│  │  • Check root flags: -sS found                          │   │
│  │  • Check sudo: NOT FOUND ❌                             │   │
│  │  • Score: 70/100                                        │   │
│  │  • Status: RECOVERABLE 🔧                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Decision: Root privilege error → Skip LLM → RECOVERABLE       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: SELF-CORRECTION                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Attempt 1/2                                             │   │
│  │  → Error: "Flags -sS require root privileges"          │   │
│  │  → Fix Strategy: Add 'sudo' prefix                     │   │
│  │  → Intent Check: "fragmentation" → Add '-f' flag       │   │
│  │  → Corrected: sudo nmap -f -sS -D 10.0.0.1,10.0.0.2... │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Re-validation:                                                 │
│   • Semantic Score: 100/100                                     │
│   • Status: VALID ✅                                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: SANDBOX TEST (Docker)                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Docker Isolation                                        │   │
│  │  1. Create nmap-sandbox network                        │   │
│  │  2. Launch nginx:alpine target container               │   │
│  │  3. Run: sudo nmap -f -sS -D RND:10 <target_ip>        │   │
│  │  4. Capture output & errors                            │   │
│  │  5. Cleanup containers                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Result:                                                        │
│   • Execution time: 12.03s                                      │
│   • Status: PASSED ✅                                           │
│   • Output: "Starting Nmap 7.98..."                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: VM EXECUTION (Ubuntu SSH)                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SSH Connection to Ubuntu VM                             │   │
│  │  • Host: 192.168.188.128:22                            │   │
│  │  • User: sdoumi                                        │   │
│  │  • Execute: sudo nmap -f -sS -D RND:10 192.168.188.128 │   │
│  │  • Save results to /home/pentester/scan-results/       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Result:                                                        │
│   • Status: SUCCESS ✅                                          │
│   • Ports found: 22/tcp open (SSH)                             │
│   • Scan completed in 15.42s                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  FINAL REPORT                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Status: SUCCESS ✅                                      │   │
│  │ Original: nmap -sS -D 10.0.0.1,10.0.0.2 192.168.188.128│   │
│  │ Final: sudo nmap -f -sS -D RND:10 192.168.188.128      │   │
│  │ Validation Score: 100/100                               │   │
│  │ Corrections Applied: 1                                  │   │
│  │ Sandbox: PASSED                                         │   │
│  │ VM Execution: SUCCESS                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

---

##  Usage Examples

### Basic Usage

```bash
# Start Agent 5 server
python run_agent5.py

# Send command via API
curl -X POST http://localhost:5002/mcp/execute \
  -H "Content-Type: application/json" \
  -d '{
    "command": "nmap -sS 192.168.1.1",
    "intent": "stealth scan",
    "target": "192.168.1.1"
  }'
```

### Response Format

```json
{
  "final_status": "success",
  "command": "sudo nmap -sS 192.168.1.1",
  "timestamp": "2026-01-08T22:26:38.953256",
  "stages": {
    "validation": {
      "status": "recoverable",
      "score": 70,
      "errors": ["Flags -sS require root privileges"],
      "method": "semantic"
    },
    "self_correction": {
      "applied": true,
      "original_command": "nmap -sS 192.168.1.1",
      "corrected_command": "sudo nmap -sS 192.168.1.1",
      "attempts": 1,
      "final_score": 100
    },
    "sandbox_execution": {
      "success": true,
      "time": 12.03,
      "output": "Starting Nmap 7.98..."
    },
    "vm_execution": {
      "success": true,
      "output": "22/tcp open ssh",
      "time": 15.42
    }
  }
}
```

---

##  Security Features

1. **Dangerous Command Detection**: Blocks rm, dd, fork bombs
2. **Docker Isolation**: Sandboxed execution before VM
3. **SSH Hardening**: Secure VM communication
4. **Input Validation**: Multiple layers of checks
5. **Rate Limiting**: Prevents abuse (30 req/min)

---

## 📚 References

- Nmap Documentation: https://nmap.org/book/
- Docker Security: https://docs.docker.com/engine/security/
- LLM Validation Papers: [Research citations]
- Semantic Validation Best Practices: [Internal docs]

---

