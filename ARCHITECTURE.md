"""
NMAP-AI CORRECTED ARCHITECTURE
================================

This document describes the corrected architecture that implements
proper separation of concerns and prevents agents from bypassing the MCP server.


ARCHITECTURE FLOW
=================

User Query
    ↓
[Agent 1] RouterAgent (DECISION MAKER)
    ├─ Comprehension Check (is query valid?)
    ├─ Complexity Classification (EASY/MEDIUM/HARD)
    └─ Agent Selection Decision (RAG or DIFFUSION)
    ↓
[Agent 2/3] RAG or Diffusion (PURE GENERATORS)
    └─ Generate command ONLY (no decisions, no validation, no execution)
    ↓
[Agent 5] MCP Server (CENTRAL AUTHORITY)
    ├─ Validation (Hybrid Semantic + LLM)
    ├─ Auto-Correction Loop (with retries)
    ├─ Sandbox Test (Docker container)
    ├─ VM Execution (SSH to Ubuntu target)
    └─ Structured Report Generation
    ↓
Response to User


KEY PRINCIPLES
==============

✅ CORRECT:
  • RouterAgent decides which agent to use (RAG or Diffusion)
  • RAG/Diffusion only generate commands
  • MCP Agent 5 is the ONLY executor
  • All commands go through MCP validation
  • Auto-correction only happens in MCP
  • Sandbox testing only happens in MCP
  • VM execution only happens in MCP

❌ WRONG (Anti-patterns):
  • Agent making its own Complexity decision
  • Agent calling MCP directly without routing
  • Agent validating its own output
  • Agent executing commands
  • Multiple agents with decision-making
  • Bypassing MCP for any reason


COMPONENT RESPONSIBILITIES
===========================

Agent 1: RouterAgent (agent_1_router/run_router.py)
  INPUT:  User query, Target IP
  OUTPUT: Decision (RAG vs Diffusion), Selected agent
  
  Steps:
    1. Comprehension check (is query about nmap?)
    2. Complexity classification (EASY/MEDIUM/HARD)
    3. Agent selection (RAG for EASY, Diffusion for MEDIUM/HARD)
    4. Call selected agent's generate() method
    5. Send generated command to MCP Agent 5

Agent 2: RAGAgent (RAG/agent/rag_agent.py)
  INPUT:  User query
  OUTPUT: Generated nmap command
  
  RULES:
    • Pure generator - no decisions
    • No validation or correction
    • No execution
    • Returns command only
    • Called ONLY by RouterAgent

Agent 3: DiffusionAgent (diffusion_models/diffusion_mcp_client.py)
  INPUT:  User query
  OUTPUT: Generated nmap command
  
  RULES:
    • Pure generator - no decisions
    • No complexity classification
    • No validation or correction
    • No execution
    • Returns command only
    • Called ONLY by RouterAgent

Agent 5: MCP Server (agent_5_validation/mcp_tools/mcp_server.py)
  INPUT:  Command, Intent, Target IP, Agent name
  OUTPUT: Structured execution report
  
  RULES:
    • Central authority for all execution
    • Validates commands (hybrid semantic + LLM)
    • Auto-corrects invalid commands (with loop)
    • Runs sandbox tests (Docker)
    • Executes on target VM (SSH)
    • Returns final status and report
    
  Pipeline:
    1. VALIDATION: Semantic checks + LLM judge
    2. AUTO-CORRECTION: Loop up to N times
    3. SANDBOX TEST: Docker container
    4. VM EXECUTION: SSH to target
    5. REPORT: Structured JSON response


MESSAGE FLOW
============

RouterAgent → RAG/Diffusion:
  {
    query: str,
    target: str
  }
  
  Response:
  {
    command: str  # The generated nmap command
  }

RouterAgent/Agent → MCP Agent 5:
  {
    command: str,
    intent: str,
    target: str,
    agent_name: str,  # "rag" or "diffusion"
    skip_sandbox: bool
  }
  
  Response:
  {
    final_status: str,  # "success", "failed_validation", "failed_vm", etc.
    command: str,       # Final command used
    stages: {
      validation: {...},
      self_correction: {...},
      sandbox: {...},
      vm_execution: {...}
    },
    timestamp: str
  }


FILES MODIFIED / CREATED
=========================

CREATED:
  ✓ run_router_main.py
    Main entry point using RouterAgent
    Use this to run the entire system

  ✓ agent_1_router/run_router.py (COMPLETELY REWRITTEN)
    New RouterAgent class
    Handles routing logic

MODIFIED:
  ✓ diffusion_models/diffusion_mcp_client.py
    Removed: ComplexityClient calls
    Removed: Complexity decision-making
    Changed: DiffusionAgent is now pure generator
    Method: generate(query) → str

  ✓ RAG/agent/rag_agent.py
    Added: async generate(query) → str method
    Already: Pure generator (no changes needed to core logic)

  ✓ agent_5_validation/mcp_tools/mcp_server.py
    Minor: Ensured final_status is always initialized
    Minor: Better error handling


RUNNING THE SYSTEM
==================

1. Start MCP Agent 5 (required first):
   python agent_5_validation/run_agent5.py

2. Start Complexity API (required):
   python agent_1_router/complexity.py
   OR run the API server that exposes it

3. (Optional) Start RAG and Diffusion services if running as servers
   These can run as either:
   - Direct imports in RouterAgent
   - Separate HTTP services

4. Start RouterAgent (new main entry point):
   python run_router_main.py


EXAMPLE FLOW
============

User: "scan ports 80 and 443 on the target"

1. RouterAgent receives query
   - Comprehension: ✓ Valid nmap query
   - Complexity: "Medium"
   - Decision: Use DIFFUSION agent

2. RouterAgent calls Diffusion.generate()
   - Returns: "nmap -p 80,443 TARGET"

3. RouterAgent sends to MCP Agent 5
   {
     command: "nmap -p 80,443 TARGET",
     intent: "scan ports 80 and 443",
     target: "192.168.188.128",
     agent_name: "diffusion"
   }

4. MCP Agent 5 executes pipeline:
   - Validation: Score 95/100 ✓
   - Auto-Correction: Not needed
   - Sandbox Test: PASSED
   - VM Execution: SUCCESS

5. Returns structured report:
   {
     final_status: "success",
     command: "nmap -p 80,443 192.168.188.128",
     stages: {
       validation: {...},
       sandbox: {...},
       vm_execution: {...}
     }
   }


TESTING
=======

Test RouterAgent only:
  python run_router_main.py

Test Diffusion Agent directly (generator):
  python diffusion_models/diffusion_mcp_client.py

Test RAG Agent directly (generator):
  python RAG/agent/rag_agent.py

Test MCP Server directly:
  curl -X POST http://localhost:5000/mcp/execute \\
    -d '{"command": "nmap -sT TARGET", "intent": "test", "target": "127.0.0.1"}'


TROUBLESHOOTING
===============

Q: "Agent bypassing MCP"
A: Make sure agent only has generate() method, not process_query()
   All execution decisions happen in RouterAgent → MCP flow

Q: "No complexity decision being made"
A: Check that RouterAgent calls ComplexityAgent.classify()
   Not the other way around

Q: "Commands not being corrected"
A: Make sure commands go to MCP Agent 5
   Auto-correction only happens in MCP, not in generation agents

Q: "Two agents generating commands in parallel"
A: RouterAgent should call ONE agent based on Complexity decision
   Not all agents

Q: "MCP not receiving proper format"
A: Check MCPExecuteRequest model:
   - command: str ✓
   - intent: str ✓
   - target: str ✓
   - agent_name: str ✓
   - skip_sandbox: bool ✓


GOLDEN RULE
===========

🚀 ONLY RouterAgent decides
🚀 ONLY RAG/Diffusion generate  
🚀 ONLY MCP Agent 5 validates, corrects, and executes
🚀 NEVER bypass MCP
🚀 NEVER make decisions in generation agents
"""
