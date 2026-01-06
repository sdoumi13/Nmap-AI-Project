#!/usr/bin/env python3
"""
Quick Test Script - Verify everything is working
"""

import subprocess
import sys
from pathlib import Path

def run_test(name, command, is_background=False):
    """Run a test command"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"Command: {command}")
    print(f"{'='*70}")
    
    try:
        if is_background:
            print(f"✅ Starting in background...")
            return True
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ SUCCESS")
                if result.stdout:
                    print(f"Output: {result.stdout[:200]}")
                return True
            else:
                print(f"❌ FAILED")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}")
                return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT (command took too long)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                      QUICK TEST SUITE                                ║
║               Verify NMAP-AI Configuration                           ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Test 1: Python version
    results["Python"] = run_test(
        "Python Version",
        "python --version"
    )
    
    # Test 2: Check if paramiko is installed
    results["Paramiko"] = run_test(
        "Paramiko Module",
        "python -c \"import paramiko; print('paramiko version:', paramiko.__version__)\""
    )
    
    # Test 3: Check if uvicorn is installed
    results["Uvicorn"] = run_test(
        "Uvicorn Module",
        "python -c \"import uvicorn; print('uvicorn version:', uvicorn.__version__)\""
    )
    
    # Test 4: Check project structure
    print(f"\n{'='*70}")
    print("TEST: Project Structure")
    print(f"{'='*70}")
    required_files = [
        "direct_vm_executor.py",
        "agent_5_validation/run_agent5.py",
        "agent_5_validation/mcp_tools/mcp_server.py",
        "agent_5_validation/execution/vm_executor.py",
        "agent_1_router/run_router.py",
        "agent_1_router/complexity.py",
        "RAG/agent/rag_agent.py",
    ]
    
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    results["Project Structure"] = all_exist
    
    # Test 5: VM Connectivity Test
    print(f"\n{'='*70}")
    print("TEST: VM Connectivity (ping 192.168.188.128)")
    print(f"{'='*70}")
    
    # Try to ping - works on Windows, Linux, Mac
    ping_cmd = "ping -n 1 192.168.188.128" if sys.platform == "win32" else "ping -c 1 192.168.188.128"
    try:
        result = subprocess.run(ping_cmd, shell=True, capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✅ VM is reachable (ping success)")
            results["VM Ping"] = True
        else:
            print("❌ VM is not reachable (ping failed)")
            print("   Make sure 192.168.188.128 is up and accessible")
            results["VM Ping"] = False
    except:
        print("⚠️ Ping test skipped")
        results["VM Ping"] = None
    
    # Test 6: Config files
    print(f"\n{'='*70}")
    print("TEST: Configuration Files")
    print(f"{'='*70}")
    
    config_files = [
        "agent_5_validation/agent5_config.yaml",
    ]
    
    all_configs = True
    for file in config_files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
        if not exists:
            all_configs = False
    
    results["Config Files"] = all_configs
    
    # Test 7: Direct VM Executor Test (if VM is accessible)
    if results.get("VM Ping"):
        print(f"\n{'='*70}")
        print("TEST: Direct VM Executor (Testing SSH connection)")
        print(f"{'='*70}")
        
        test_cmd = "python direct_vm_executor.py -c \"whoami\""
        try:
            result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and "sdoumi" in result.stdout:
                print("✅ SSH connection successful (can reach VM)")
                results["VM SSH"] = True
            else:
                print("❌ SSH connection failed")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}")
                results["VM SSH"] = False
        except subprocess.TimeoutExpired:
            print("⏱️ SSH test timed out")
            results["VM SSH"] = False
        except Exception as e:
            print(f"❌ Error: {e}")
            results["VM SSH"] = False
    
    # Summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name}")
            passed += 1
        elif result is False:
            print(f"❌ {test_name}")
            failed += 1
        else:
            print(f"⚠️ {test_name} (skipped)")
            skipped += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*70}")
    
    if failed == 0:
        print("\n✅ ALL CRITICAL TESTS PASSED! You're ready to use the system.")
        print("\nNext steps:")
        print("  1. Run: python direct_vm_executor.py -c \"nmap -sV localhost\" --sudo")
        print("  2. Or: python run_router_main.py (with other services running)")
    else:
        print(f"\n❌ {failed} test(s) failed. Fix them before proceeding.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
