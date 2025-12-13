# test_ssh.py
from execution.vm_executor import VMExecutor

vm_config = {
    'host': '192.168.188.128',
    'username': 'sdoumi',  # Change ici
    'password': 'kali',   # ou use key_file
    'port': 22
}

print("Testing SSH connection to Ubuntu VM...")

try:
    with VMExecutor(vm_config) as vm:
        # Test simple command
        result = vm.execute(
            command="nmap --version",
            target=""
        )
        
        if result['success']:
            print(" SSH connection successful!")
            print(f"Nmap version:\n{result['output']}")
        else:
            print(" SSH failed!")
            print(f"Errors: {result['errors']}")

except Exception as e:
    print(f" Connection error: {e}")