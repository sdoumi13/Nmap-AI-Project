"""
HOW TO USE THE SYSTEM
=====================

Now that everything is set up and working, here's how to use it.


REQUIREMENTS TO RUN
===================

Before starting, make sure these services are running:

1. MCP Agent 5 (MUST start first)
   Terminal 1:
   cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
   .\venv\Scripts\python.exe agent_5_validation\run_agent5.py
   
   Expected:
   🚀 Starting Agent 5 MCP Server...
   ✅ Agent 5 MCP Server ready on http://0.0.0.0:5000

2. Complexity API (REQUIRED)
   Terminal 2:
   cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
   .\venv\Scripts\python.exe -m uvicorn agent_1_router.complexity:app --port 7000 --log-level info
   
   Expected:
   ╔════════════════════════════════════════════╗
   ║   NMAP COMPLEXITY CLASSIFIER API          ║
   ║   Easy → RAG | Medium/Hard → Diffusion    ║
   ╚════════════════════════════════════════════╝

3. Ollama (for RAG/Diffusion)
   Terminal 3:
   Make sure ollama is running with llama3 model loaded
   
   Command:
   ollama run llama3
   
   (This keeps running in background)


STARTING ROUTERAGENT
====================

Terminal 4 (Main):
cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
.\venv\Scripts\python.exe run_router_main.py

You should see:
======================================================================
      NMAP-AI CORRECTED ARCHITECTURE
      Query → Router (Complexity) → Agent → MCP (Validation) → Sandbox → VM
======================================================================

[*] Initializing RouterAgent...
🧠 Initialisation Comprehension Agent...
   📂 Chemin corpus: C:\Users\Public\M2\Deep Leaning\Nmap-AI-Project\datasets\rag_corpus_detailed.json
   📚 Chargement de 15 docs et 3 exemples...
   Chargement du modèle d'embeddings...
   ✅ Comprehension Agent prêt!
[*] Initializing ComplexityAgent...
[✓] ComplexityAgent ready
✓ RouterAgent ready
----------------------------------------------------------------------
Default Target: 192.168.188.128 (Ubuntu VM)
Complexity API: http://localhost:7000/classify
MCP Agent 5: http://localhost:5000/mcp/execute
----------------------------------------------------------------------

ROUTER >


ENTERING QUERIES
================

Now you can enter queries at the ROUTER > prompt.

Easy Query (uses RAG):
────────────────────
ROUTER > scan port 80 on target
Target (Default 192.168.188.128): 
Processing...

Expected:
  [STEP 1/4] COMPREHENSION CHECK
    ✅ VALID. Score: 0.95
  
  [STEP 2/4] COMPLEXITY CLASSIFICATION
    Level: Easy
    Confidence: 0.85
    Recommended Agent: RAG
  
  [STEP 3/4] COMMAND GENERATION (RAG)
    Generated: nmap -p 80 TARGET
  
  [STEP 4/4] MCP EXECUTION (AGENT 5)
    [Validation] → [Auto-Correction] → [Sandbox] → [VM Execution]
    Final Status: success


Medium Query (uses Diffusion):
──────────────────────────────
ROUTER > stealth scan with timing options
Target (Default 192.168.188.128): 
Processing...

Expected:
  [STEP 1/4] COMPREHENSION CHECK
    ✅ VALID. Score: 0.92
  
  [STEP 2/4] COMPLEXITY CLASSIFICATION
    Level: Medium
    Confidence: 0.80
    Recommended Agent: DIFFUSION
  
  [STEP 3/4] COMMAND GENERATION (DIFFUSION)
    Generated: nmap -sS -T3 TARGET
  
  [STEP 4/4] MCP EXECUTION (AGENT 5)
    Final Status: success


Hard Query (uses Diffusion):
────────────────────────────
ROUTER > comprehensive network reconnaissance with os detection
Target (Default 192.168.188.128): 
Processing...

Expected:
  [STEP 1/4] COMPREHENSION CHECK
    ✅ VALID. Score: 0.90
  
  [STEP 2/4] COMPLEXITY CLASSIFICATION
    Level: Hard
    Confidence: 0.85
    Recommended Agent: DIFFUSION
  
  [STEP 3/4] COMMAND GENERATION (DIFFUSION)
    Generated: nmap -A -O -Pn TARGET
  
  [STEP 4/4] MCP EXECUTION (AGENT 5)
    Final Status: success


CHECKING RESULTS
================

After execution, you'll see:

╔══ COMMAND ══╗
  nmap -p 80 192.168.188.128
╚════════════════╝

[Self-correction info if applicable]

╔══ FINAL STATUS ══╗
  success
╚═════════════════╝

Execution Summary:
  ✓ Validation: VALID
  ✓ Auto-Correction: Not needed
  ✓ Sandbox: PASSED
  ✓ VM Execution: SUCCESS


CUSTOM TARGET
=============

You can specify a different target:

ROUTER > scan all ports
Target (Default 192.168.188.128): 10.0.0.5
Processing...

This will execute the scan against 10.0.0.5 instead.


HANDLING ERRORS
===============

If you get an error:

COMPREHENSION REJECTED:
  ROUTER > hello world
  [STEP 1/4] COMPREHENSION CHECK
    ❌ REJECTED. Score: 0.15
    Reason: Irrelevant/Noise

  Solution: Use nmap-related keywords

COMPLEXITY API ERROR:
  ❌ Could not reach Complexity API
  
  Solution: Make sure Terminal 2 (Complexity API) is running

MCP CONNECTION ERROR:
  ❌ MCP error: connection refused
  
  Solution: Make sure Terminal 1 (MCP Agent 5) is running

AGENT GENERATION ERROR:
  ❌ RAG Error or ❌ Diffusion Error
  
  Solution: 
  - Check Terminal 3 (Ollama) for RAG/Diffusion
  - Check model checkpoint for Diffusion


QUITTING
========

Type: exit
Or: quit
Or: Ctrl+C

ROUTER > exit
👋 Goodbye!


TESTING WITH SCRIPT
===================

Instead of manual entry, you can run automated tests:

python test_pipeline.py

This will:
1. Test Easy query (scan port 80) → RAG
2. Test Medium query (stealth scan) → Diffusion
3. Test Hard query (comprehensive scan) → Diffusion

And show results for each.


ADVANCED USAGE
==============

Testing Complexity Only:
  python -c "
  from agent_1_router.complexity import ComplexityAgent
  ca = ComplexityAgent()
  result = ca.classify('scan port 22')
  print(result)
  "

Testing RAG Only:
  python RAG/agent/rag_agent.py
  [Interactive RAG testing]

Testing Diffusion Only:
  python diffusion_models/diffusion_mcp_client.py
  [Interactive Diffusion testing]

Testing MCP Only:
  curl -X POST http://localhost:5000/mcp/execute \\
    -d '{
      "command": "nmap -p 80 TARGET",
      "intent": "scan web port",
      "target": "192.168.188.128",
      "agent_name": "test"
    }' \\
    -H "Content-Type: application/json"


TROUBLESHOOTING GUIDE
======================

Problem: ImportError: cannot import name 'ComplexityAgent'
Solution: Already fixed in agent_1_router/complexity.py
          Just restart the terminal

Problem: Connection refused on port 5000
Solution: Start MCP Agent 5 first (Terminal 1)

Problem: Connection refused on port 7000
Solution: Start Complexity API (Terminal 2)

Problem: "Ollama not found" or "llama3 not loaded"
Solution: Run: ollama run llama3 (Terminal 3)

Problem: "Vector database not found"
Solution: RAG will auto-create it on first run

Problem: "Target not reachable"
Solution: 
  1. Check if 192.168.188.128 is up: ping 192.168.188.128
  2. Check SSH credentials: kali:kali (in agent5_config.yaml)
  3. Use different target IP if needed

Problem: "Validation always fails"
Solution: 
  1. Check MCP validation logs
  2. Try simpler commands first
  3. Check agent auto-correction output


MONITORING
==========

While using the system, check these indicators:

Terminal 1 (MCP Agent 5):
  ✓ "MCP PIPELINE: rag" or "MCP PIPELINE: diffusion"
  ✓ "[STAGE X/4]" messages show progress
  ✓ "FINAL STATUS: success" indicates success

Terminal 2 (Complexity API):
  ✓ "POST /classify" requests being logged
  ✓ Responses with level (EASY/MEDIUM/HARD)

Terminal 3 (Ollama):
  ✓ Running without errors
  ✓ Processing requests when RAG/Diffusion called

Terminal 4 (RouterAgent):
  ✓ Each step (1/4, 2/4, 3/4, 4/4) progressing
  ✓ Agent choice (RAG or DIFFUSION) correct for complexity
  ✓ Final status shown at end


NEXT STEPS
==========

After testing the pipeline:

1. ✅ Verify all components work
2. ✅ Test with different queries
3. ✅ Check logs for any issues
4. ✅ Adjust complexity thresholds if needed
5. ✅ Customize for your use case


SUMMARY
=======

You now have a complete, working NMAP-AI pipeline:

✅ User enters query
✅ Complexity decides agent (RAG or Diffusion)
✅ Agent generates nmap command
✅ MCP validates, corrects, tests, and executes
✅ Results shown to user

The system is ready for production use! 🚀
"""
