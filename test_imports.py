#!/usr/bin/env python3
"""
Quick test to verify all imports work correctly
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")

try:
    from agent_1_router.complexity import ComplexityAgent
    print("✅ ComplexityAgent imported successfully")
    
    ca = ComplexityAgent()
    result = ca.classify("scan port 80")
    print(f"✅ ComplexityAgent.classify() works: {result['level']}")
    
except Exception as e:
    print(f"❌ Error with ComplexityAgent: {e}")
    import traceback
    traceback.print_exc()

try:
    from agent_1_router.comprehension import ComprehensionAgent
    print("✅ ComprehensionAgent imported successfully")
except Exception as e:
    print(f"❌ Error with ComprehensionAgent: {e}")

try:
    from agent_1_router.run_router import RouterAgent
    print("✅ RouterAgent imported successfully")
except Exception as e:
    print(f"❌ Error with RouterAgent: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All imports successful!")
