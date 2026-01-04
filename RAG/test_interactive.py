import os
import sys
import time

# Import de ton module RAG
from agent.rag_agent import NmapRagAgent

# Couleurs pour le terminal (Style Kali Linux)
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def type_effect(text):
    """Petit effet visuel pour afficher le texte comme un terminal rétro"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)
    print("")

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{BLUE}{BOLD}" + "="*60)
    print("      NMAP RAG AGENT - INTERACTIVE SHELL v1.0")
    print("      Architecture: Local (Ollama/Llama3 + ChromaDB)")
    print("="*60 + f"{RESET}\n")

    # 1. Chargement
    if not os.path.exists("nmap_dataset.json"):
        print(f"{RED}[!] Erreur critique : nmap_dataset.json introuvable !{RESET}")
        return

    print(f"{YELLOW}[*] Initialisation du moteur RAG... Veuillez patienter.{RESET}")
    try:
        # On initialise l'agent
        agent = NmapRagAgent(dataset_path="nmap_dataset.json")
        print(f"{GREEN}[V] Agent RAG chargé et prêt !{RESET}")
    except Exception as e:
        print(f"{RED}[X] Erreur au démarrage de l'agent : {e}{RESET}")
        print(f"{YELLOW}Astuce : Vérifie que 'ollama run llama3' tourne bien.{RESET}")
        return

    print("-" * 60)
    print("Instructions :")
    print(" - Tape ta demande en langage naturel (Français ou Anglais)")
    print(" - Tape 'exit' ou 'quit' pour fermer le programme")
    print("-" * 60)

    # 2. Boucle infinie pour le chat
    while True:
        try:
            user_input = input(f"\n{BLUE}{BOLD}Pentester@{os.environ.get('USERNAME', 'Kali')} > {RESET}")
            
            # Gestion de la sortie
            if user_input.lower() in ['exit', 'quit']:
                print(f"\n{YELLOW}[*] Fermeture de la session.{RESET}")
                break
            
            if not user_input.strip():
                continue

            # Traitement
            start_time = time.time()
            print(f"{YELLOW} ... Analyse et Génération ...{RESET}", end="\r")
            
            # Appel à ton agent
            response = agent.process({"user_query": user_input})
            
            duration = time.time() - start_time

            # Affichage du résultat
            if response.get("status") == "success":
                cmd = response.get("nmap_candidate")
                print(" " * 40, end="\r") # Efface le message de chargement
                
                print(f"{GREEN}➜ Commande Générée ({duration:.2f}s) :{RESET}")
                print(f"{BOLD}    {cmd}{RESET}")
                
            else:
                err = response.get("error_message")
                print(f"\n{RED}[!] Erreur : {err}{RESET}")

        except KeyboardInterrupt:
            print(f"\n{YELLOW}[*] Interruption détectée. Au revoir.{RESET}")
            break

if __name__ == "__main__":
    main()