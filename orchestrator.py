"""
NMAP-AI ORCHESTRATOR
Démarre tous les agents dans le bon ordre avec MCP activé
"""

import subprocess
import time
import os
import sys
import signal
import requests
from typing import List, Dict
from pathlib import Path

# Couleurs
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ProcessManager:
    """Gestionnaire de processus pour tous les agents"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
    
    def start(self, name: str, command: List[str], cwd: str = ".") -> bool:
        """Démarre un processus"""
        try:
            print(f"{YELLOW}[*] Démarrage {name}...{RESET}")
            
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[name] = process
            print(f"{GREEN}[✓] {name} démarré (PID: {process.pid}){RESET}")
            return True
        
        except Exception as e:
            print(f"{RED}[✗] Échec {name}: {e}{RESET}")
            return False
    
    def stop_all(self):
        """Arrête tous les processus"""
        print(f"\n{YELLOW}[*] Arrêt de tous les services...{RESET}")
        
        for name, process in self.processes.items():
            try:
                print(f"  - Arrêt {name} (PID: {process.pid})")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"    Erreur: {e}")
        
        print(f"{GREEN}[✓] Tous les services arrêtés{RESET}")
    
    def check_health(self, name: str, url: str, max_retries: int = 10) -> bool:
        """Vérifie si un service est prêt"""
        print(f"{YELLOW}  Attente de {name}...{RESET}", end="")
        
        for i in range(max_retries):
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    print(f" {GREEN}OK{RESET}")
                    return True
            except:
                pass
            
            print(".", end="", flush=True)
            time.sleep(1)
        
        print(f" {RED}TIMEOUT{RESET}")
        return False


def check_prerequisites():
    """Vérifie que tout est installé"""
    print(f"\n{BLUE}{BOLD}{'='*70}")
    print("  NMAP-AI ORCHESTRATOR - VÉRIFICATION PRÉALABLE")
    print(f"{'='*70}{RESET}\n")
    
    checks = {
        "Python": sys.version_info >= (3, 8),
        "agent5_config.yaml": Path("agent5_config.yaml").exists(),
        "nmap_dataset.json": Path("nmap_dataset.json").exists(),
    }
    
    all_ok = True
    for name, status in checks.items():
        icon = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
        print(f"  {icon} {name}")
        all_ok = all_ok and status
    
    if not all_ok:
        print(f"\n{RED}[!] Des prérequis manquent. Arrêt.{RESET}")
        sys.exit(1)
    
    print(f"\n{GREEN}[✓] Tous les prérequis OK{RESET}\n")


def start_all_services(manager: ProcessManager):
    """Démarre tous les services dans le bon ordre"""
    
    print(f"{BLUE}{BOLD}{'='*70}")
    print("  DÉMARRAGE DES SERVICES")
    print(f"{'='*70}{RESET}\n")
    
    services = [
        {
            "name": "Agent 5 - MCP Server",
            "command": ["python", "agent_5_validation_self_correction/run_agent_5.py"],
            "cwd": ".",
            "url": "http://localhost:5000",
            "critical": True
        },
        {
            "name": "Agent 3 - Diffusion API",
            "command": ["python", "agent_3_diffusion/run_diffusion.py"],
            "cwd": ".",
            "url": "http://localhost:8000",
            "critical": False
        },
        {
            "name": "Agent 2 - RAG",
            "command": ["python", "agent_2_rag/run_rag.py", "--mcp"],
            "cwd": ".",
            "url": None,  # Pas de serveur, juste client
            "critical": False
        }
    ]
    
    # Démarrage séquentiel
    for service in services:
        success = manager.start(
            name=service["name"],
            command=service["command"],
            cwd=service["cwd"]
        )
        
        if not success and service["critical"]:
            print(f"\n{RED}[!] Service critique '{service['name']}' échoué. Arrêt.{RESET}")
            manager.stop_all()
            sys.exit(1)
        
        # Health check si URL fournie
        if service["url"]:
            if not manager.check_health(service["name"], service["url"]):
                if service["critical"]:
                    print(f"\n{RED}[!] Service critique non accessible. Arrêt.{RESET}")
                    manager.stop_all()
                    sys.exit(1)
        
        time.sleep(2)  # Pause entre chaque service
    
    print(f"\n{GREEN}{BOLD}[✓] TOUS LES SERVICES SONT DÉMARRÉS{RESET}\n")


def show_service_urls():
    """Affiche les URLs des services"""
    print(f"{BLUE}{'='*70}")
    print("  SERVICES DISPONIBLES")
    print(f"{'='*70}{RESET}")
    print(f"""
  {BOLD}Agent 5 - MCP Server{RESET}
    • Validation: http://localhost:5000/mcp/validate
    • Exécution:  http://localhost:5000/mcp/execute
    • Health:     http://localhost:5000/health
  
  {BOLD}Agent 3 - Diffusion API{RESET}
    • Generate:   http://localhost:8000/generate
    • Execute:    http://localhost:8000/execute
    • Health:     http://localhost:8000/health
  
  {BOLD}Agent 2 - RAG{RESET}
    • Mode interactif (CLI)
    • Utilise MCP Client pour communiquer avec Agent 5
  
  {YELLOW}Documentation complète:{RESET} http://localhost:5000/docs
    """)
    print(f"{BLUE}{'='*70}{RESET}\n")


def show_examples():
    """Affiche des exemples d'utilisation"""
    print(f"{BLUE}{'='*70}")
    print("  EXEMPLES D'UTILISATION")
    print(f"{'='*70}{RESET}")
    print(f"""
  {BOLD}1. Test rapide de validation:{RESET}
    curl -X POST http://localhost:5000/mcp/validate \\
      -H "Content-Type: application/json" \\
      -d '{{"command": "nmap -sT -p 80,443 TARGET", 
           "intent": "scan web ports",
           "agent_name": "test"}}'
  
  {BOLD}2. Génération via Diffusion:{RESET}
    curl -X POST http://localhost:8000/generate \\
      -H "Content-Type: application/json" \\
      -d '{{"query": "scan all ports on target"}}'
  
  {BOLD}3. Pipeline complet:{RESET}
    curl -X POST http://localhost:8000/execute \\
      -H "Content-Type: application/json" \\
      -d '{{"query": "stealth scan web ports",
           "target": "scanme.nmap.org"}}'
  
  {BOLD}4. RAG interactif:{RESET}
    python agent_2_rag/run_rag.py
    > scan web services on target
    """)
    print(f"{BLUE}{'='*70}{RESET}\n")


def interactive_menu(manager: ProcessManager):
    """Menu interactif de contrôle"""
    while True:
        print(f"\n{BOLD}OPTIONS:{RESET}")
        print("  [1] Voir les URLs des services")
        print("  [2] Voir les exemples d'utilisation")
        print("  [3] Test rapide (validation)")
        print("  [4] Vérifier le statut")
        print("  [q] Quitter et arrêter tous les services")
        
        choice = input(f"\n{BLUE}Choix > {RESET}").strip().lower()
        
        if choice == '1':
            show_service_urls()
        
        elif choice == '2':
            show_examples()
        
        elif choice == '3':
            test_command = input("Commande à tester: ").strip()
            if test_command:
                try:
                    response = requests.post(
                        "http://localhost:5000/test/quick-validate",
                        params={
                            "command": test_command,
                            "intent": "test command"
                        },
                        timeout=10
                    )
                    result = response.json()
                    
                    print(f"\n{BOLD}Résultat:{RESET}")
                    print(f"  Status: {result.get('status')}")
                    print(f"  Score: {result.get('score')}/100")
                    if result.get('errors'):
                        print(f"  Erreurs: {result.get('errors')}")
                
                except Exception as e:
                    print(f"{RED}Erreur: {e}{RESET}")
        
        elif choice == '4':
            print(f"\n{BOLD}Statut des services:{RESET}")
            services = [
                ("Agent 5", "http://localhost:5000/health"),
                ("Agent 3", "http://localhost:8000/health")
            ]
            
            for name, url in services:
                try:
                    r = requests.get(url, timeout=2)
                    status = f"{GREEN}ONLINE{RESET}" if r.status_code == 200 else f"{RED}ERROR{RESET}"
                except:
                    status = f"{RED}OFFLINE{RESET}"
                
                print(f"  {name}: {status}")
        
        elif choice == 'q':
            print(f"\n{YELLOW}[*] Arrêt demandé{RESET}")
            break


def main():
    """Point d'entrée principal"""
    
    # Configuration pour capturer Ctrl+C proprement
    manager = ProcessManager()
    
    def signal_handler(sig, frame):
        print(f"\n{YELLOW}[*] Signal d'interruption reçu{RESET}")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Étape 1 : Vérifications
        check_prerequisites()
        
        # Étape 2 : Démarrage
        start_all_services(manager)
        
        # Étape 3 : Affichage info
        show_service_urls()
        show_examples()
        
        # Étape 4 : Menu interactif
        print(f"{GREEN}{BOLD}[✓] Système prêt ! Utilisation du menu interactif...{RESET}\n")
        interactive_menu(manager)
    
    except Exception as e:
        print(f"\n{RED}[!] Erreur fatale : {e}{RESET}")
        import traceback
        traceback.print_exc()
    
    finally:
        manager.stop_all()
        print(f"\n{BLUE}Au revoir !{RESET}")


if __name__ == "__main__":
    main()