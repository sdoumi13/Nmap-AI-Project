#!/usr/bin/env python3
"""
Test the complete RouterAgent pipeline
This tests: Query → Complexity → Agent → MCP
"""

import sys
import asyncio
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent_1_router.run_router import RouterAgent

async def test_pipeline():
    """Test complete pipeline with sample queries"""
    
    print("="*70)
    print("TESTING COMPLETE NMAP-AI PIPELINE")
    print("Query → Complexity → Agent → MCP → Validation → Sandbox → VM")
    print("="*70)
    
    router = RouterAgent(
        complexity_url="http://localhost:7000",
        mcp_url="http://localhost:5000"
    )
    
    # Test queries
    test_queries = [
        ("scan port 80 on target", "192.168.188.128"),  # Easy - should use RAG
        ("stealth scan with timing", "192.168.188.128"),  # Medium - should use Diffusion
        ("comprehensive network reconnaissance", "192.168.188.128"),  # Hard - should use Diffusion
    ]
    
    for query, target in test_queries:
        print(f"\n{'='*70}")
        print(f"TEST QUERY: {query}")
        print(f"TARGET: {target}")
        print("="*70)
        
        try:
            result = await router.route(query, target)
            
            if result["status"] == "rejected":
                print(f"\n❌ REJECTED: {result['reason']}")
            elif result["status"] == "generation_failed":
                print(f"\n❌ GENERATION FAILED: {result.get('agent', 'unknown')} agent")
            elif result["status"] == "executed":
                print(f"\n✅ EXECUTION SUCCESSFUL")
                print(f"   Agent Used: {result.get('agent')}")
                print(f"   Command Generated: {result.get('command_generated')}")
                
                exec_result = result.get('execution', {})
                final_status = exec_result.get('final_status', 'unknown')
                print(f"   Final Status: {final_status}")
                
                if final_status == 'success':
                    print("   ✅ All stages passed!")
                else:
                    print(f"   ⚠️  Some stages failed or skipped")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    await router.close()
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_pipeline())
