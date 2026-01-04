"""
FIX: COMPLEXITY AGENT CIRCULAR IMPORT
======================================

PROBLEM
=======

When running: python agent_1_router/run_router.py

Error:
  ImportError: cannot import name 'ComplexityAgent' from 'agent_1_router.complexity'

Root Cause:
  The file agent_1_router/complexity.py was trying to import ComplexityAgent from itself:
  
  from agent_1_router.complexity import ComplexityAgent  ← Circular import!
  
  This file IS complexity.py, but it was importing from complexity.py, creating a circular reference.


SOLUTION
========

Moved ComplexityAgent class definition INTO complexity.py instead of trying to import it.

The ComplexityAgent class now:
  1. Classifies queries into Easy/Medium/Hard complexity levels
  2. Uses keyword matching against complexity-specific keyword lists
  3. Returns: level, confidence, and reasoning
  4. Is instantiated by the FastAPI app when it starts up


IMPLEMENTATION
==============

File: agent_1_router/complexity.py

Added:
  • ComplexityAgent class with:
    - __init__(): Initializes keyword lists for each complexity level
    - classify(query): Analyzes query and returns complexity classification
  
  • Keyword lists:
    - easy_keywords: Basic scan operations (port scan, host discovery, etc.)
    - medium_keywords: Intermediate techniques (stealth, timing, scripts, etc.)
    - hard_keywords: Advanced operations (OS detection, exploits, bruteforce, etc.)

Removed:
  • Circular import: from agent_1_router.complexity import ComplexityAgent


VERIFICATION
============

To test:
  python test_imports.py
  
Expected output:
  ✅ ComplexityAgent imported successfully
  ✅ ComplexityAgent.classify() works: Easy
  ✅ ComprehensionAgent imported successfully
  ✅ RouterAgent imported successfully
  ✅ All imports successful!

To run RouterAgent:
  python run_router_main.py
  
Or from agent_1_router directory:
  python run_router.py


BEHAVIOR
========

ComplexityAgent Classification:

Input: "scan port 80"
  → Matches 2 easy keywords
  → Result: {level: 'Easy', confidence: 0.67, reason: 'Query contains basic scanning keywords'}

Input: "stealth scan with firewall evasion"
  → Matches 3 medium keywords
  → Result: {level: 'Medium', confidence: 0.75, reason: 'Query contains intermediate keywords'}

Input: "comprehensive os detection and vulnerability scanning"
  → Matches 3 hard keywords
  → Result: {level: 'Hard', confidence: 0.75, reason: 'Query contains advanced keywords'}

Input: "hello world"
  → No matches
  → Result: {level: 'Medium', confidence: 0.5, reason: 'Default classification'}


ROUTING DECISION
================

RouterAgent uses ComplexityAgent classification to route to the correct agent:

complexity['level'] == 'Easy'
  → Route to RAG Agent (faster, uses vector search)

complexity['level'] in ('Medium', 'Hard')
  → Route to Diffusion Agent (more creative, handles complex scenarios)


FILES CHANGED
=============

✓ agent_1_router/complexity.py
  - Removed circular import
  - Added ComplexityAgent class definition
  - Now self-contained and importable

✓ test_imports.py [NEW]
  - Added test script to verify all imports work
  - Tests ComplexityAgent functionality
  - Provides quick validation


NEXT STEPS
==========

1. Run test: python test_imports.py
2. Run RouterAgent: python run_router_main.py
3. Or run as API server: python -m uvicorn agent_1_router.complexity:app --port 7000


NOTES
=====

• ComplexityAgent is now self-contained in complexity.py
• Can be used by RouterAgent directly (import in run_router.py)
• Can also be run as FastAPI server (when if __name__ == "__main__" is executed)
• Uses simple keyword matching (can be enhanced with ML models later)
• Confidence scores are based on keyword matches (0.0 to 1.0)
"""
