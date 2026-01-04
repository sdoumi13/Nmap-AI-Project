
"""
Step 7: Sandbox Executor (Docker)
Safe testing environment before VM execution
"""

import docker
from typing import Dict

class SandboxExecutor:
    """Docker-based sandbox for safe testing"""
    
    def __init__(self):
        self.client = docker.from_env()
        self._setup_network()
    
    def _setup_network(self):
        """Create isolated network"""
        try:
            self.network = self.client.networks.get("nmap-sandbox")
        except docker.errors.NotFound:
            self.network = self.client.networks.create(
                "nmap-sandbox",
                driver="bridge",
                internal=True
            )
    
    async def execute(self, command: str, timeout: int = 60) -> Dict:
        """
        Execute command in Docker sandbox
        Returns: {success: bool, output: str, errors: List, time: float}
        """
        # Create target container
        target = self.client.containers.run(
            "nginx:alpine",
            detach=True,
            network="nmap-sandbox",
            remove=True
        )
        
        try:
            # Get target IP
            target.reload()
            target_ip = target.attrs['NetworkSettings']['Networks']['nmap-sandbox']['IPAddress']
            
            # Replace TARGET
            safe_cmd = command.replace('TARGET', target_ip)
            
            # Run Nmap
            result = self.client.containers.run(
                "instrumentisto/nmap:latest",
                command=safe_cmd,
                network="nmap-sandbox",
                remove=True,
                stdout=True,
                stderr=True
            )
            
            output = result.decode()
            
            return {
                "success": True,
                "output": output,
                "errors": [],
                "time": self._parse_time(output)
            }
        
        except docker.errors.ContainerError as e:
            return {
                "success": False,
                "output": "",
                "errors": [str(e)],
                "time": 0
            }
        
        finally:
            try:
                target.stop()
                target.remove()
            except:
                pass
    
    def _parse_time(self, output: str) -> float:
        """Extract execution time from output"""
        import re
        match = re.search(r'in ([\d.]+) seconds', output)
        return float(match.group(1)) if match else 0.0
