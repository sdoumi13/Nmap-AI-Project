
"""
Step 8: VM Executor
Execute validated command in real VM (final step)
"""

import paramiko
from typing import Dict

class VMExecutor:
    """Execute commands in VM via SSH"""
    
    def __init__(self, vm_config: Dict):
        self.config = vm_config
        self.ssh = None
    
    def connect(self):
        """Establish SSH connection"""
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        self.ssh.connect(
            hostname=self.config['host'],
            port=self.config.get('port', 22),
            username=self.config['username'],
            password=self.config.get('password'),
            key_filename=self.config.get('key_file')
        )
    
    def execute(self, command: str, target: str) -> Dict:
        """
        Execute nmap command in VM
        
        Args:
            command: Validated nmap command
            target: Actual target IP/domain
        
        Returns: {success: bool, output: str, errors: List}
        """
        if not self.ssh:
            self.connect()
        
        # Replace TARGET placeholder with actual target
        final_command = command.replace('TARGET', target)
        
        try:
            # Execute command
            stdin, stdout, stderr = self.ssh.exec_command(
                final_command,
                timeout=300  # 5 minutes max
            )
            
            output = stdout.read().decode()
            errors = stderr.read().decode()
            exit_code = stdout.channel.recv_exit_status()
            
            return {
                "success": (exit_code == 0),
                "output": output,
                "errors": [errors] if errors else [],
                "exit_code": exit_code
            }
        
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "errors": [str(e)],
                "exit_code": -1
            }
    
    def disconnect(self):
        """Close SSH connection"""
        if self.ssh:
            self.ssh.close()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

