# test_docker.py
import asyncio
from execution.sandbox_executor import SandboxExecutor

async def test_docker():
    print("Testing Docker sandbox...")
    
    sandbox = SandboxExecutor()
    
    result = await sandbox.execute("nmap -sT -p 80,443 TARGET", timeout=60)
    
    if result['success']:
        print(" Docker sandbox working!")
        print(f"Execution time: {result['time']}s")
        print(f"Output preview:\n{result['output'][:300]}")
    else:
        print(" Docker sandbox failed!")
        print(f"Errors: {result['errors']}")

if __name__ == "__main__":
    asyncio.run(test_docker())