# Agent 5: Validation & Self-Correction Architecture

## MCP Protocol 

### Overview of the MCP Protocol

The **MCP (Model Context Protocol)** is the communication backbone used to securely and consistently exchange data between **Agent 1 (Router)** and **Agent 5 (Validation & Execution Agent)**.

In the Nmap-AI system, MCP acts as a **control and governance layer** that ensures every generated Nmap command is:
- Semantically valid
- Security-compliant
- Safely executable
- Fully traceable

MCP is **not a simple API call**, but a structured protocol that defines:
- Message format
- Validation states
- Execution permissions
- Feedback loops

---

![Validation & Self-Correction Flow](/Annexe//validation+self-correction.png)

---

##  System Architecture

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

##  Validation Pipeline

### 1. Semantic Validator (Rule-Based)

**Purpose**: Fast, deterministic validation using regex patterns and semantic rules.

**Key Features**:
-  Detects dangerous commands (rm -rf, fork bombs)
-  Validates root privilege requirements
-  Checks target presence (IP/domain)
-  Identifies flag conflicts (-sS vs -sT)
-  Validates port syntax

---

### 2. LLM Judge (Qwen2.5-Coder-3B / Mistral-7B)
![LLM](/Annexe//LLM.png)

**Purpose**: Intelligent validation using language models for complex cases.

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

**Decision Matrix**:
| Semantic Score | LLM Score | Final Status |
|----------------|-----------|--------------|
| 100            | -         | VALID      |
| 70 (root err)  | -         | RECOVERABLE 🔧 |
| < 50           | -         | INVALID ❌   |
| 60-79          | 80-100    | VALID      |
| 60-79          | 50-79     | RECOVERABLE 🔧 |

---
![Validation](/Annexe//Validation.png)

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
           ├─► VALID (score ≥ 75)     →  Success
           ├─► Still Invalid          →  Retry (max 2)
           └─► Max Retries            →  Return best attempt
```

---

##  Security Features

1. **Dangerous Command Detection**: Blocks rm, dd, fork bombs
2. **Docker Isolation**: Sandboxed execution before VM
3. **SSH Hardening**: Secure VM communication
4. **Input Validation**: Multiple layers of checks
5. **Rate Limiting**: Prevents abuse (30 req/min)

---

> **MCP transforms LLM-generated commands into controlled, auditable, and secure operations.**  
It is the cornerstone that makes Nmap-AI suitable for **real-world cybersecurity environments**.

##  References

- Nmap Documentation: https://nmap.org/book/
- Docker Security: https://docs.docker.com/engine/security/
- LLM Validation Papers
- Semantic Validation 

---

