# Nmap-AI-Project

"""
NMAP-AI Agent 5 - Complete Implementation
Validation + Self-Correction + MCP + VM Execution

Project Structure:
agent5/
├── __init__.py
├── validation/
│   ├── __init__.py
│   ├── semantic_validator.py      # Step 1: Semantic rules
│   ├── llm_judge.py                # Step 2: LLM fallback
│   └── hybrid_validator.py         # Step 3: Combined validation
├── mcp/
│   ├── __init__.py
│   ├── protocol.py                 # Step 4: MCP message schemas
│   ├── server.py                   # Step 5: MCP server
│   └── client.py                   # Step 6: MCP client
├── execution/
│   ├── __init__.py
│   ├── sandbox_executor.py         # Step 7: Docker sandbox
│   └── vm_executor.py              # Step 8: VM execution
├── self_correction/
│   ├── __init__.py
│   └── corrector.py                # Step 9: Self-correction loop
└── orchestrator.py                 # Step 10: Main orchestrator
"""

