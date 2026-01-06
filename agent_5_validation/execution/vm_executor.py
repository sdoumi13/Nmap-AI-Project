
"""
Step 8: VM Executor
Execute validated command in real VM (final step)
"""

import paramiko
import time
from typing import Dict

class VMExecutor:
    """Execute commands in VM via SSH"""
    
    def __init__(self, vm_config: Dict):
        self.config = vm_config
        self.ssh = None
        self.password = vm_config.get('password', '')
    
    def connect(self):
        """Establish SSH connection"""
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        self.ssh.connect(
            hostname=self.config['host'],
            port=self.config.get('port', 22),
            username=self.config['username'],
            password=self.config.get('password'),
            key_filename=self.config.get('key_file'),
            timeout=10
        )
    
    def execute(self, command: str, target: str) -> Dict:
        """
        Execute nmap command in VM with automatic sudo password handling
        
        Args:
            command: Validated nmap command
            target: Actual target IP/domain
        
        Returns: {success: bool, output: str, errors: List, exit_code: int}
        """
        if not self.ssh:
            self.connect()
        
        # Replace TARGET placeholder with actual target
        final_command = command.replace('TARGET', target)
        
        try:
            # If command contains sudo, we need to handle password input
            if final_command.startswith('sudo'):
                # Use sudo with -S flag to read password from stdin
                # This allows non-interactive password input
                final_command = f"echo '{self.password}' | sudo -S {final_command[5:].strip()}"
            
            # Execute command with timeout
            stdin, stdout, stderr = self.ssh.exec_command(
                final_command,
                timeout=self.config.get('command_timeout', 300)  # Default 5 minutes
            )
            
            # Read output
            output = stdout.read().decode('utf-8', errors='ignore')
            error_output = stderr.read().decode('utf-8', errors='ignore')
            exit_code = stdout.channel.recv_exit_status()
            
            # Detect success
            is_success = (exit_code == 0) and ('sudo: a terminal is required' not in error_output)
            
            return {
                "success": is_success,
                "output": output,
                "errors": [error_output] if error_output and not is_success else [],
                "exit_code": exit_code,
                "command_executed": final_command
            }
        
        except paramiko.ssh_exception.SSHException as ssh_err:
            return {
                "success": False,
                "output": "",
                "errors": [f"SSH Error: {str(ssh_err)}"],
                "exit_code": -1,
                "command_executed": final_command
            }
        
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "errors": [f"Execution Error: {str(e)}"],
                "exit_code": -1,
                "command_executed": final_command
            }
    
    def disconnect(self):
        """Close SSH connection"""
        if self.ssh:
            try:
                self.ssh.close()
            except:
                pass
            self.ssh = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

