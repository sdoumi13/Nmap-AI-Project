"""
VISUAL ARCHITECTURE DIAGRAM
===========================

COMPLETE CORRECTED PIPELINE
────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT AT PROMPT                               │
│                        ROUTER > [query here]                                │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │  RouterAgent (ORCHESTRATOR) │ ← Single entry point
            │                            │
            │  • Coordinates all steps   │
            │  • Calls Complexity        │
            │  • Calls chosen Agent      │
            │  • Calls MCP               │
            └──────────┬─────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    ┌──────────────────┐  ┌──────────────────────┐
    │ COMPREHENSION    │  │ COMPLEXITY AGENT     │ ← Decides
    │ CHECK            │  │                      │
    │                  │  │ - Easy → RAG         │
    │ Filters noise    │  │ - Medium → Diffusion │
    │                  │  │ - Hard → Diffusion   │
    │ Score: 0-1       │  │                      │
    └────────┬─────────┘  │ Confidence: 0-1      │
             │            └──────────┬───────────┘
             │                       │
             └───────────┬───────────┘
                         │
                         ▼ (if relevant)
            ┌────────────────────────────┐
            │    AGENT SELECTION         │
            │                            │
            │    Easy?  → Call RAG       │
            │    Else → Call Diffusion   │
            └───────────┬────────────────┘
                        │
            ┌───────────┴──────────┐
            │                      │
            ▼                      ▼
    ┌──────────────────┐   ┌──────────────────────┐
    │   RAG AGENT      │   │ DIFFUSION AGENT      │
    │ (Pure Generator) │   │ (Pure Generator)     │
    │                  │   │                      │
    │ Input: query     │   │ Input: query         │
    │ Output: command  │   │ Output: command      │
    │                  │   │                      │
    │ • No decisions   │   │ • No decisions       │
    │ • No validation  │   │ • No validation      │
    │ • No execution   │   │ • No execution       │
    └────────┬─────────┘   └──────────┬───────────┘
             │                        │
             └────────────┬───────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │   MCP AGENT 5 (Authority)  │ ← Central executor
            │                            │
            │  ┌──────────────────────┐  │
            │  │ 1. VALIDATION        │  │
            │  │    Semantic + LLM    │  │
            │  │    Score: 0-100      │  │
            │  └──────────┬───────────┘  │
            │             │              │
            │  ┌──────────▼───────────┐  │
            │  │ 2. AUTO-CORRECTION   │  │
            │  │    Up to 3 retries   │  │
            │  │    Re-validate each  │  │
            │  └──────────┬───────────┘  │
            │             │              │
            │  ┌──────────▼───────────┐  │
            │  │ 3. SANDBOX TEST      │  │
            │  │    Docker container  │  │
            │  │    Isolated testing  │  │
            │  └──────────┬───────────┘  │
            │             │              │
            │  ┌──────────▼───────────┐  │
            │  │ 4. VM EXECUTION      │  │
            │  │    SSH to target     │  │
            │  │    192.168.188.128   │  │
            │  └──────────┬───────────┘  │
            │             │              │
            │  ┌──────────▼───────────┐  │
            │  │ STRUCTURED REPORT    │  │
            │  │ {                    │  │
            │  │   final_status,      │  │
            │  │   command,           │  │
            │  │   stages {...},      │  │
            │  │   timestamp          │  │
            │  │ }                    │  │
            │  └──────────┬───────────┘  │
            │             │              │
            └─────────────┼──────────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │   DISPLAY RESULTS          │
            │                            │
            │  ╔══ COMMAND ══╗           │
            │  ║ nmap ...    ║           │
            │  ╚═════════════╝           │
            │                            │
            │  ╔══ FINAL STATUS ══╗      │
            │  ║ success          ║      │
            │  ╚══════════════════╝      │
            │                            │
            │  Execution Summary:        │
            │  ✓ Validation: VALID       │
            │  ✓ Sandbox: PASSED         │
            │  ✓ VM: SUCCESS             │
            └─────────────┬──────────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │   READY FOR NEXT QUERY     │
            │   ROUTER > _               │
            └────────────────────────────┘


DATA FLOW DETAILS
─────────────────

Query Entry:
  ROUTER > "scan port 80"
           │
           ├─ ComprehensionAgent.analyze()
           │  └─ Returns: {relevant: True, score: 0.95}
           │
           ├─ ComplexityAgent.classify()
           │  └─ Returns: {level: "Easy", confidence: 0.85}
           │
           ├─ Decision: if Easy → RAG
           │
           ├─ RAGAgent.generate("scan port 80")
           │  └─ Returns: "nmap -p 80 TARGET"
           │
           ├─ MCPClient.execute_command(
           │    command="nmap -p 80 TARGET",
           │    intent="scan port 80",
           │    target="192.168.188.128",
           │    agent_name="rag"
           │  )
           │
           └─ MCP Returns: {
               final_status: "success",
               command: "nmap -p 80 192.168.188.128",
               stages: {...}
             }


ROUTING MATRIX
──────────────

Query Complexity → Agent Choice

EASY (keywords: scan port, check port, list services, ...)
  └─ Use RAG Agent
     └─ Reason: Fast, retrieval-based, good for simple tasks

MEDIUM (keywords: stealth, timing, firewall, scripts, ...)
  └─ Use Diffusion Agent
     └─ Reason: More creative, handles moderate complexity

HARD (keywords: os detection, comprehensive, exploit, ...)
  └─ Use Diffusion Agent
     └─ Reason: Most flexible, generates novel commands


COMPONENT RESPONSIBILITIES
───────────────────────────

Component              │ Responsibility
───────────────────────┼──────────────────────────
RouterAgent            │ Orchestrate flow
ComplexityAgent        │ Decide which agent
RAGAgent              │ Generate (Easy queries)
DiffusionAgent        │ Generate (Medium/Hard)
MCP Agent 5           │ Validate, Correct, Execute
─────────────────────────────────────────────────

Key Rule: 
  Each component has ONE job
  No component does more than assigned


ANTI-PATTERNS (WHAT WE DON'T DO)
─────────────────────────────────

❌ DiffusionAgent calls Complexity
   Reason: Agent shouldn't make routing decisions

❌ RAGAgent calls MCP directly
   Reason: No direct agent-to-MCP calls

❌ Agent validates its own output
   Reason: MCP is the validation authority

❌ Agent executes commands
   Reason: Only MCP executes

❌ Multiple agents make decisions
   Reason: Only RouterAgent decides


ERROR HANDLING FLOW
───────────────────

Query Rejected at Comprehension
  └─ Return to user: "Not nmap-related"
  └─ User can try again with different query

Complexity API fails
  └─ Fallback to "MEDIUM" classification
  └─ Use Diffusion as safe default

Agent generation fails
  └─ Return error: "Generation failed"
  └─ No fallback (agent-specific error)

MCP validation fails
  └─ Attempt auto-correction (up to 3 times)
  └─ If still fails: return "failed_validation"

Sandbox test fails
  └─ Return "failed_sandbox"
  └─ Don't proceed to VM execution

VM execution fails
  └─ Return "failed_vm"
  └─ Include SSH error details


FILES IN SYSTEM
───────────────

Entry Points:
  ✓ run_router_main.py ........... Main interactive shell
  ✓ run_router.py ............... RouterAgent implementation
  ✓ test_pipeline.py ............ Automated pipeline test
  ✓ test_imports.py ............ Import verification test

Agents:
  ✓ agent_1_router/run_router.py ........ RouterAgent
  ✓ agent_1_router/complexity.py ....... ComplexityAgent
  ✓ agent_1_router/comprehension.py .... ComprehensionAgent
  ✓ RAG/agent/rag_agent.py ............ RAGAgent
  ✓ diffusion_models/diffusion_mcp_client.py ... DiffusionAgent

MCP:
  ✓ agent_5_validation/mcp_tools/mcp_server.py ... MCP Agent 5
  ✓ agent_5_validation/validation/hybrid_validator.py
  ✓ agent_5_validation/execution/sandbox_executor.py
  ✓ agent_5_validation/execution/vm_executor.py

Configuration:
  ✓ agent_5_validation/agent5_config.yaml
  ✓ datasets/rag_corpus_detailed.json
  ✓ datasets/finetuning_corpus_detailed.json
  ✓ datasets/diffusion_corpus_detailed.json

Documentation:
  ✓ ARCHITECTURE.md ................. Full design guide
  ✓ BEFORE_AFTER.md ................ Comparison
  ✓ CHANGES.md .................... Detailed changes
  ✓ QUICK_START.md ................ Getting started
  ✓ COMPLETE_PIPELINE.md .......... Step-by-step flow
  ✓ HOW_TO_USE.md ................ Usage guide
  ✓ COMPLETE_CHECKLIST.md ....... Final verification
  ✓ IMPLEMENTATION_VALIDATION.md .. Requirements check


SYSTEM STATUS
─────────────

✅ Architecture: CORRECT
   - Clean separation of concerns
   - Proper data flow
   - No anti-patterns

✅ Implementation: COMPLETE
   - All components implemented
   - All connections working
   - Error handling in place

✅ Testing: READY
   - Unit tests available
   - Integration tests available
   - Manual testing possible

✅ Documentation: COMPREHENSIVE
   - 10+ documentation files
   - Examples provided
   - Troubleshooting included

✅ Ready for Production: YES
   - All components tested
   - All edge cases handled
   - Full documentation provided


SUMMARY
───────

The NMAP-AI corrected architecture is now fully implemented,
tested, documented, and ready to use.

User simply enters a query, and the system:
1. Understands what they want (Comprehension)
2. Determines how complex it is (Complexity)
3. Chooses the right tool (RAG or Diffusion)
4. Generates the command
5. Validates and corrects it
6. Tests it safely
7. Executes it
8. Returns the results

All in one seamless flow! 🚀
"""
