# Flux MCP Simplifié - Visualisation Rapide

## 📊 Diagramme 1: Vue d'ensemble du système

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          NMAP-AI DISTRIBUTED SYSTEM                        │
└────────────────────────────────────────────────────────────────────────────┘

                         YOUR MACHINE (192.168.1.169)
                    ┌─────────────────────────────────┐
                    │  USER INPUT: "scan tcp port 22" │
                    └────────────┬────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────┐
                    │   ROUTER AGENT (port 7000)      │
                    │  - Comprehension Check ✓        │
                    │  - Complexity Classification    │
                    │    └─→ EASY (confidence: 0.95) │
                    └────────────┬────────────────────┘
                                 │
                    EASY?   YES   │
                                 ▼
                    ┌─────────────────────────────────┐
                    │   USE DISTRIBUTED RAG           │
                    │  (Colleague's Machine)          │
                    │  http://192.168.1.218:8000      │
                    └────────────┬────────────────────┘
                                 │
                                 │ HTTP POST /generate_command
                                 │
                                 ▼
        COLLEAGUE MACHINE (192.168.1.218)
        ┌─────────────────────────────────┐
        │  AGENT 2 SERVER (port 8000)     │
        │  ┌───────────────────────────┐  │
        │  │ NmapRagAgent.process()    │  │
        │  │ 1. ChromaDB search        │  │
        │  │    → Find similar commands│  │
        │  │ 2. Ollama generation      │  │
        │  │    → Generate command     │  │
        │  │ 3. Return result          │  │
        │  └───────────────┬───────────┘  │
        │                  │               │
        │                  ▼               │
        │  Generated: "nmap -sT -p 22 ..." │
        │                  │               │
        │  ┌───────────────▼───────────┐  │
        │  │ MCPAgent5Client.validate()│  │
        │  │ HTTP POST validate        │  │
        │  │ http://192.168.1.169:5000 │  │
        │  └───────────────┬───────────┘  │
        └────────────────┬────────────────┘
                         │
                         │ HTTP POST /mcp/validate
                         │ {command, intent, target}
                         │
                         ▼
        YOUR MACHINE (192.168.1.169)
        ┌─────────────────────────────────┐
        │  AGENT 5 MCP (port 5000)        │
        │  ┌───────────────────────────┐  │
        │  │ HybridValidator           │  │
        │  │ - Semantic validation     │  │
        │  │ - LLM validation (if<80)  │  │
        │  │ - Final score: 97/100     │  │
        │  └───────────────┬───────────┘  │
        │                  │               │
        │                  ▼               │
        │  {valid: true, score: 97}       │
        │                  │               │
        │                  │ HTTP 200 OK   │
        │                  │               │
        │  ┌───────────────▼───────────┐  │
        │  │ Back to Colleague RAG     │  │
        │  └───────────────┬───────────┘  │
        └────────────────┬────────────────┘
                         │
        COLLEAGUE MACHINE (192.168.1.218)
        ┌────────────────▼────────────────┐
        │  Received validation            │
        │  Include in response:           │
        │  {                              │
        │    command: "nmap -sT -p 22",  │
        │    validation: {                │
        │      valid: true,               │
        │      score: 97,                 │
        │      method_used: "semantic"    │
        │    }                            │
        │  }                              │
        │                                 │
        │  HTTP 200 OK back to Router     │
        └────────────┬────────────────────┘
                     │
                     │ HTTP 200 OK
                     │ command + validation
                     │
                     ▼
        YOUR MACHINE (192.168.1.169)
        ┌─────────────────────────────────┐
        │  ROUTER receives response       │
        │                                 │
        │  ✅ Command: nmap -sT -p 22...  │
        │     Validated: Score 97/100     │
        │     (hybrid)                    │
        │                                 │
        │  Display to user + Send to      │
        │  Agent 5 for execution          │
        └────────────┬────────────────────┘
                     │
                     │ HTTP POST /mcp/execute
                     │
                     ▼
        ┌─────────────────────────────────┐
        │  AGENT 5 FULL PIPELINE          │
        │  1. Validation ✓                │
        │  2. Self-Correction             │
        │  3. Sandbox (Docker) ✓          │
        │  4. VM (SSH) ✓                  │
        │                                 │
        │  Result: SUCCESS                │
        │  Port 22/tcp open ssh           │
        └────────────┬────────────────────┘
                     │
                     ▼
                ┌──────────────┐
                │  FINAL RESULT│
                │  SUCCESS ✓   │
                └──────────────┘
```

---

## 📈 Diagramme 2: Flux de données

```
┌─────────────┐
│ User Query  │ "scan tcp port 22"
└──────┬──────┘
       │
       ▼
   ┌─────────────────────────────┐
   │ ComplexityAgent             │
   │ .classify(query)            │
   └──────┬──────────────────────┘
          │
          ├─ Keywords matching: ['tcp', 'port', 'scan']
          ├─ Score: EASY (1.0)
          ├─ Confidence: 0.95
          │
          ▼
      ╔═════════════════════════╗
      ║ Decision: USE RAG (EASY)║
      ╚═════════╤═══════════════╝
                │
                ▼
    ┌────────────────────────────────────────┐
    │ DistributedRAGClient                   │
    │ POST http://192.168.1.218:8000         │
    │ /generate_command                      │
    │ {query, target, source_agent}          │
    └────────┬─────────────────────────────┬─┘
             │                             │
      [Colleague Machine]            [Your Machine]
      192.168.1.218:8000             192.168.1.169
             │                             │
             ▼                             │
    ┌────────────────────────┐            │
    │ NmapRagAgent           │            │
    │ .process(query, target)│            │
    └────────┬───────────────┘            │
             │                             │
             ├─ ChromaDB search:           │
             │  "scan tcp port 22"         │
             │  → "nmap -sT -p 22 ..." ✓   │
             │  (similarity: 0.97)         │
             │                             │
             ├─ Ollama generation:         │
             │  "nmap -sT -p 22 192.168..." │
             │  (adapted to target) ✓      │
             │                             │
             ├─ Result:                   │
             │  command: "nmap -sT..."     │
             │  confidence: 0.92           │
             │                             │
             ├─ Validate via MCP:         │
             │  POST /mcp/validate         │
             │  on Your Agent 5            │
             │         │                   │
             │         └──────────────────→│
             │                             ▼
             │              ┌──────────────────────────┐
             │              │ Agent 5 Validator        │
             │              │ .validate_semantic()     │
             │              │ .validate_llm() (if<80)  │
             │              │                          │
             │              │ Result:                  │
             │              │ valid: true              │
             │              │ score: 97/100            │
             │              │ method: semantic         │
             │              └──────────┬───────────────┘
             │                         │
             │                    HTTP 200
             │←────────────────────────┘
             │
             └─ Receive validation:
             │  score: 97/100 ✓
             │
             ├─ Build response:
             │  {
             │    status: "success",
             │    command: "nmap -sT -p 22...",
             │    validation: {
             │      valid: true,
             │      score: 97,
             │      method_used: "semantic"
             │    }
             │  }
             │
             ▼
    HTTP 200 OK
    Back to Router
    ├─ command
    ├─ validation score
    └─ metadata
             │
             │ [Back to Your Machine]
             │
             ▼
    ┌──────────────────────────────────┐
    │ Router receives validated command │
    │                                  │
    │ ✅ Command valid (score: 97)    │
    │                                  │
    │ → Send to Agent 5 for execution  │
    └────────┬─────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────┐
    │ Agent 5 Full Execution Pipeline   │
    │                                  │
    │ 1. Validate ✓                    │
    │ 2. Self-Correct (skip)           │
    │ 3. Sandbox (Docker) ✓            │
    │ 4. VM (SSH) ✓                    │
    │                                  │
    │ Result: Port 22 open             │
    └────────┬─────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────┐
    │ Final Response to User           │
    │                                  │
    │ Status: SUCCESS ✓                │
    │ Port 22: ssh (OPEN)              │
    │ Time: 5.04s                      │
    └──────────────────────────────────┘
```

---

## 🔀 Diagramme 3: Routing basé sur complexité

```
┌─────────────────────┐
│ User Query          │
└────────┬────────────┘
         │
         ▼
    ┌─────────────────────────────┐
    │ Complexity Classification    │
    └──────────┬──────────────────┘
               │
        ┌──────┼──────┬──────┐
        │      │      │      │
        ▼      ▼      ▼      ▼
      EASY  MEDIUM  HARD  DEFAULT
        │      │      │      │
        │      │      │      └─→ Medium (fallback)
        │      │      │
        │      │      └─→ Diffusion (Complex model)
        │      │          - Generates advanced commands
        │      │          - Used for hard queries
        │      │
        │      └─→ Diffusion (Generative model)
        │          - Generates varied commands
        │          - Used for medium complexity
        │
        └─→ RAG (Retrieval-Augmented Generation)
            - Fast & accurate
            - Uses examples from dataset
            - Cross-network to colleague
            - Validated via MCP

Example Keywords:
├─ EASY:
│  ├─ "scan port"
│  ├─ "tcp port"
│  ├─ "host scan"
│  └─ "check service"
│
├─ MEDIUM:
│  ├─ "stealth"
│  ├─ "firewall"
│  ├─ "version detection"
│  └─ "ipv6"
│
└─ HARD:
   ├─ "vulnerability"
   ├─ "exploit"
   ├─ "os detection"
   └─ "brute force"
```

---

## ⏱️ Diagramme 4: Timing

```
User Input
   │
   ├─→ [0.05s] Comprehension Check
   │
   ├─→ [0.02s] Complexity Classification
   │
   ├─→ [0.50s] Network: Query to Colleague RAG
   │   │
   │   ├─→ [0.10s] ChromaDB Search
   │   ├─→ [0.20s] Ollama Generation
   │   └─→ [0.15s] MCP Validation (sync)
   │       │
   │       └─→ [0.05s] Agent 5 Validation
   │       └─→ [0.10s] Network latency
   │
   ├─→ [0.05s] Network: Response back to Router
   │
   ├─→ [2.34s] Sandbox Execution (Docker)
   │
   ├─→ [2.12s] VM Execution (SSH)
   │
   └─→ [0.01s] Format & return response
      ────────
      TOTAL: ~5.04 seconds

Breakdown:
├─ Network: 0.65s (13%)
├─ Processing: 0.07s (1%)
├─ Execution: 4.46s (86%)
└─ Other: 0.04s (~1%)
```

---

## 🔗 Diagramme 5: Architecture réseau

```
Internet/LAN
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│ YOUR NETWORK             │    │ COLLEAGUE'S NETWORK      │
│ 192.168.1.169            │    │ 192.168.1.218            │
│                          │    │                          │
│ ┌────────────────────┐   │    │ ┌────────────────────┐   │
│ │ Router (7000)      │───┼────┼→│ Agent 2 (8000)     │   │
│ │                    │   │    │ │                    │   │
│ │ Agent 5 (5000)     │←──┼────┼─│ MCP Client         │   │
│ │ - Validator        │   │    │ │ (calls back)       │   │
│ │ - Sandbox          │   │    │ │                    │   │
│ │ - VM Executor      │   │    │ │ NmapRagAgent       │   │
│ │                    │   │    │ │ - ChromaDB         │   │
│ │ Ollama (11434)     │   │    │ │ - Ollama           │   │
│ │ - Llama3           │   │    │ │                    │   │
│ │ - Mistral          │   │    │ │ Ollama (11434)     │   │
│ └────────────────────┘   │    │ │ - Llama3           │   │
│                          │    │ │ - Mistral          │   │
└──────────────────────────┘    │ └────────────────────┘   │
                                 │                          │
                                 └──────────────────────────┘
                                      │
                                      │ SSH :22
                                      │
                                      ▼
                                  ┌──────────────────────────┐
                                  │ TARGET VM (192.168.188.128)
                                  │ Ubuntu                   │
                                  │ - NMAP execution target  │
                                  └──────────────────────────┘

Key Points:
├─ REST API: Router → Agent 2 (HTTP, port 8000)
├─ MCP Protocol: Agent 2 → Agent 5 (HTTP, port 5000)
├─ Local MCP: Router → Agent 5 (HTTP, port 5000)
├─ SSH: Agent 5 → VM (port 22)
└─ Ollama: Local (port 11434)
```

---

## 📋 Diagramme 6: Classe et dépendances

```
Router (run_router.py)
    │
    ├─ ComplexityAgent (complexity.py)
    │  └─ classify(query) → Easy/Medium/Hard
    │
    ├─ ComprehensionAgent (comprehension.py)
    │  └─ analyze(query) → {relevant, score}
    │
    ├─ DistributedRAGClient (distributed_routing.py)
    │  └─ generate_command(query, target)
    │     └─ HTTP POST http://192.168.1.218:8000
    │
    └─ MCPClient (run_router.py)
       ├─ validate_command(...)
       │  └─ HTTP POST http://localhost:5000/mcp/validate
       │
       └─ execute_command(...)
          └─ HTTP POST http://localhost:5000/mcp/execute

Agent 5 MCP Server (mcp_server.py)
    │
    ├─ Agent5MCPServer
    │  │
    │  ├─ /mcp/validate endpoint
    │  │  └─ HybridValidator
    │  │     ├─ semantic_validator.py
    │  │     └─ llm_judge.py (Mistral)
    │  │
    │  ├─ /mcp/execute endpoint
    │  │  ├─ HybridValidator
    │  │  ├─ SelfCorrectionAgent
    │  │  ├─ SandboxExecutor (Docker)
    │  │  └─ VMExecutor (SSH)
    │  │
    │  └─ /health endpoint
    │
    └─ Supporting services
       ├─ ChromaDB (vectorial search)
       ├─ Ollama (embeddings + generation)
       └─ Docker (sandbox execution)

NmapRagAgent (rag_agent.py - Colleague)
    │
    ├─ ChromaDB Client
    │  └─ collection.query(...) → Similar commands
    │
    ├─ Ollama Client
    │  └─ generate(model="llama3", prompt=...) → Command
    │
    └─ MCPAgent5ClientSync
       └─ validate_command(...) → {valid, score}
          └─ HTTP POST http://192.168.1.169:5000
```

---

## 📚 Légende

```
→  HTTP/API call
←  HTTP/API response
┌─ Start
┘  End
├─ Branch/Option
│  Continuation
▼  Next step
✓  Success/Completed
✗  Error/Failed
●  Current state
```

---

## 🎯 Points clés

```
✓ EASY queries:   RAG (Colleague) + MCP Validation (You)
✓ MEDIUM/HARD:   Diffusion (You) + MCP Validation (You)
✓ Execution:     Always through Agent 5 Full Pipeline
✓ Validation:    Hybrid (Semantic + LLM)
✓ Safety:        Sandbox + VM isolation
✓ Speed:         Cache + Parallel processing
```

---

C'est tout! Vous avez une visualisation complète du flux MCP. 🚀
