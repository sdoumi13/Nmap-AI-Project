"""
COMPLETE PIPELINE FLOW GUIDE
============================

This document explains what happens when you enter a query in RouterAgent.


STEP-BY-STEP EXECUTION
======================

When you type: "scan port 80 on target"
And hit Enter at: ROUTER >


Step 1: COMPREHENSION CHECK
──────────────────────────────

RouterAgent.route() is called with:
  - user_query = "scan port 80 on target"
  - target = "192.168.188.128"

Step 1a: Comprehension Analysis
  ComprehensionAgent.analyze(query) checks:
    • Is this about nmap? (uses semantic embeddings + TF-IDF)
    • Returns: {relevant: True/False, score: 0.0-1.0, reason: str}
  
  Output:
    ✅ VALID. Score: 0.95
    
  If NOT relevant → Return "rejected" status

Step 1b: Continue to next step
  comp_result['relevant'] = True ✓
  → Continue to Step 2


Step 2: COMPLEXITY CLASSIFICATION
──────────────────────────────────

ComplexityAgent.classify(query) analyzes:
  • How complex is this query?
  • Should use RAG or Diffusion?

Keyword Matching:
  easy_keywords = ["scan port", "check port", "list services", ...]
  medium_keywords = ["stealth", "timing", "firewall", ...]
  hard_keywords = ["os detection", "comprehensive", "exploit", ...]

For "scan port 80 on target":
  • easy_keywords matched: 2 ("scan port")
  • medium_keywords matched: 0
  • hard_keywords matched: 0
  → Winner: EASY (matches most keywords in this category)

Output:
  Level: Easy
  Confidence: 0.67
  Recommended Agent: RAG

Decision:
  if level == "Easy":
    agent_choice = "RAG"
  else:
    agent_choice = "DIFFUSION"
  
  Result: agent_choice = "RAG" ✓


Step 3: COMMAND GENERATION
──────────────────────────────

RouterAgent calls the selected agent's generate() method.

For RAG Agent:
  ──────────────
  RAGAgent.generate("scan port 80 on target")
  
  Process:
    1. TF-IDF + Cosine Similarity to find matching examples
    2. Retrieves 3 most similar nmap commands from vector DB
    3. LLM (Ollama llama3) generates response using examples
    4. Returns: "nmap -p 80 TARGET"
  
  Output: command = "nmap -p 80 TARGET"

For Diffusion Agent:
  ──────────────────
  DiffusionAgent.generate("scan ports 443 and 8080")
  
  Process:
    1. Loads discrete diffusion model (T5-based)
    2. Runs diffusion sampler for 15 steps
    3. Generates command conditioned on query
    4. Returns: "nmap -p 443,8080 TARGET"
  
  Output: command = "nmap -p 443,8080 TARGET"


Step 4: MCP EXECUTION (AGENT 5)
──────────────────────────────────

RouterAgent sends generated command to MCP:
  MCPClient.execute_command(
    command="nmap -p 80 TARGET",
    intent="scan port 80 on target",
    target="192.168.188.128",
    agent_name="rag"
  )

MCP Pipeline:
  
  4a: VALIDATION
      • Semantic checks (is this a valid nmap command?)
      • LLM judge (additional validation with Mistral/Ollama)
      • Returns: valid: True/False, score: 0-100
  
  4b: AUTO-CORRECTION (if needed)
      If validation failed:
      • Loop up to 3 times
      • Auto-correct based on errors
      • Re-validate each time
  
  4c: SANDBOX TEST (Docker)
      • Run command in isolated Docker container
      • Check if it executes without errors
      • Verify output is reasonable
  
  4d: VM EXECUTION (SSH)
      • Connect to target VM (192.168.188.128)
      • Execute validated command
      • Capture output
      • Return results

Output: Complete structured report:
  {
    final_status: "success",
    command: "nmap -p 80 192.168.188.128",
    stages: {
      validation: {...},
      self_correction: {...},
      sandbox: {...},
      vm_execution: {...}
    },
    timestamp: "2026-01-04T..."
  }


COMPLETE FLOW DIAGRAM
======================

User Input: "scan port 80"
        ↓
    RouterAgent
        ├─→ [1] ComprehensionAgent.analyze()
        │        └─→ ✅ VALID (0.95 score)
        │
        ├─→ [2] ComplexityAgent.classify()
        │        └─→ EASY (confidence: 0.67)
        │
        ├─→ [3a] Route Decision
        │         └─→ complexity == EASY?
        │              YES → Use RAG Agent
        │
        ├─→ [3b] RAGAgent.generate()
        │         └─→ "nmap -p 80 TARGET"
        │
        ├─→ [4] MCPClient.execute_command()
        │        ├─→ Validation: PASSED (95/100)
        │        ├─→ Auto-Correction: Not needed
        │        ├─→ Sandbox: PASSED
        │        └─→ VM Execution: SUCCESS
        │
        └─→ Display Results to User
             ✅ Final Status: SUCCESS


EXAMPLE EXECUTION TRACE
======================

When you enter: "stealth scan with firewall evasion"

[STEP 1/4] COMPREHENSION CHECK
  ✅ VALID. Score: 0.92

[STEP 2/4] COMPLEXITY CLASSIFICATION
  Level: Medium
  Confidence: 0.80
  Recommended Agent: DIFFUSION

[STEP 3/4] COMMAND GENERATION (DIFFUSION)
  Generated: nmap -sS --spoof-mac 0 --decoy TARGET

[STEP 4/4] MCP EXECUTION (AGENT 5)

  [STAGE 1/4] VALIDATION
    Status: VALID
    Score: 92/100

  [STAGE 2/4] AUTO-CORRECTION
    Not needed (validation passed)

  [STAGE 3/4] SANDBOX TEST
    ✅ Sandbox PASSED

  [STAGE 4/4] VM EXECUTION
    ✅ VM Execution SUCCESS

╔══ COMMAND ══╗
  nmap -sS --spoof-mac 0 --decoy 192.168.188.128
╚════════════════╝

╔══ FINAL STATUS ══╗
  success
╚═════════════════╝


TESTING THE PIPELINE
====================

Quick Test:
  python test_pipeline.py
  
  This will test 3 sample queries:
  1. Easy: "scan port 80 on target"
  2. Medium: "stealth scan with timing"
  3. Hard: "comprehensive network reconnaissance"

Interactive Test:
  python run_router_main.py
  
  Then type queries at the prompt:
  ROUTER > your query here
  Target (Default 192.168.188.128): <press enter or enter IP>


DEBUGGING
=========

If a step fails, check:

1. Comprehension fails (REJECTED)
   → Query not recognized as nmap-related
   → Try more specific terms like "nmap", "scan", "port", etc.

2. ComplexityAgent returns error
   → Complexity API not running? Start: python agent_1_router/complexity.py
   → Or increase timeout in ComplexityClient

3. Agent generation fails
   → RAG: Ollama not running? Start: ollama run llama3
   → Diffusion: Model checkpoint missing? Check nmap_diffusion_checkpoint/
   
4. MCP fails
   → Agent 5 not running? Start: python agent_5_validation/run_agent5.py
   → Check connection to 192.168.188.128

5. VM Execution fails
   → Target VM not accessible
   → Check SSH credentials in agent5_config.yaml
   → Verify target is running: ping 192.168.188.128


KEY POINTS
==========

✓ RouterAgent is the ORCHESTRATOR - it doesn't generate, it routes
✓ ComplexityAgent DECIDES which agent to use
✓ RAG/Diffusion ONLY GENERATE - they don't decide or validate
✓ MCP VALIDATES, CORRECTS, and EXECUTES
✓ Each stage is separate and can fail independently
✓ If any stage fails, the pipeline stops and reports the error
✓ All information is logged and returned to the user


FILES INVOLVED
==============

agent_1_router/run_router.py       RouterAgent (orchestrator)
agent_1_router/complexity.py       ComplexityAgent (decider)
agent_1_router/comprehension.py    ComprehensionAgent (relevance filter)
RAG/agent/rag_agent.py              RAGAgent (generator)
diffusion_models/diffusion_mcp_client.py  DiffusionAgent (generator)
agent_5_validation/mcp_tools/mcp_server.py  MCP Agent 5 (validator/executor)


SUMMARY
=======

The complete pipeline is now fully implemented:

1. ✅ User enters query
2. ✅ Comprehension check filters irrelevant queries
3. ✅ Complexity agent decides RAG vs Diffusion
4. ✅ Selected agent generates nmap command
5. ✅ MCP Agent 5 validates the command
6. ✅ MCP auto-corrects if needed
7. ✅ MCP runs sandbox test
8. ✅ MCP executes on target VM
9. ✅ Full report returned to user

Ready to use! 🚀
"""
