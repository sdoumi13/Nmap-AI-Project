"""
Script pour créer la structure complète du projet avec tous les __init__.py
Exécuter depuis la racine du projet: python setup_project.py
"""

import os
from pathlib import Path

def create_init_files():
    """Crée tous les fichiers __init__.py nécessaires"""
    
    # Structure du projet
    directories = [
        # Agent 1
        "agent_1_router",
        
        # Agent 2 RAG
        "RAG",
        "RAG/agent",
        
        # Agent 3 Diffusion
        "agent_3_diffusion",
        
        # Agent 5 Validation
        "agent_5_validation",
        "agent_5_validation/validation",
        "agent_5_validation/mcp_tools",
        "agent_5_validation/execution",
        "agent_5_validation/self_correction",
    ]
    
    print("🔧 Création des fichiers __init__.py...")
    
    for directory in directories:
        dir_path = Path(directory)
        
        # Créer le dossier s'il n'existe pas
        if not dir_path.exists():
            print(f"  📁 Création du dossier: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Créer __init__.py
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            print(f"  ✅ Création de: {init_file}")
            init_file.touch()
        else:
            print(f"  ⏭️  Existe déjà: {init_file}")
    
    print("\n✅ Tous les fichiers __init__.py sont créés!\n")


def create_config_files():
    """Crée les fichiers de configuration par défaut"""
    
    print("📝 Création des fichiers de configuration...")
    
    # agent5_config.yaml
    config_content = """validation:
  mistral_api_url: "http://localhost:1234/v1/chat/completions"
  max_retries: 3

docker:
  timeout: 60

vm:
  host: "192.168.188.128"
  port: 22
  username: "kali"
  password: "kali"
  # Alternative: utiliser une clé SSH
  # key_file: "/path/to/ssh/key"
"""
    
    config_path = Path("agent_5_validation/agent5_config.yaml")
    if not config_path.exists():
        print(f"  ✅ Création de: {config_path}")
        with open(config_path, 'w') as f:
            f.write(config_content)
    else:
        print(f"  ⏭️  Existe déjà: {config_path}")
    
    print("\n✅ Fichiers de configuration créés!\n")


def install_dependencies():
    """Affiche les commandes pour installer les dépendances"""
    
    print("📦 Installation des dépendances:")
    print("\n" + "="*60)
    print("Exécutez les commandes suivantes dans votre terminal:")
    print("="*60)
    print()
    print("# 1. Activer l'environnement virtuel")
    print("venv\\Scripts\\activate  # Windows")
    print("# ou")
    print("source venv/bin/activate  # Linux/Mac")
    print()
    print("# 2. Mettre à jour pip")
    print("python -m pip install --upgrade pip")
    print()
    print("# 3. Installer les dépendances principales")
    print("pip install fastapi uvicorn pydantic httpx")
    print()
    print("# 4. Installer LangChain et ChromaDB")
    print("pip install langchain==0.1.0 langchain-community langchain-chroma chromadb")
    print()
    print("# 5. Installer les dépendances ML")
    print("pip install torch transformers sentence-transformers scikit-learn")
    print()
    print("# 6. Installer les utilities")
    print("pip install pyyaml paramiko docker")
    print()
    print("# Ou installer tout d'un coup depuis requirements.txt:")
    print("pip install -r requirements.txt")
    print()
    print("="*60)
    print()


def check_project_structure():
    """Vérifie la structure actuelle du projet"""
    
    print("🔍 Vérification de la structure du projet...\n")
    
    required_files = {
        "RAG/RAG_MCP_Client.py": "Client MCP pour RAG",
        "RAG/agent/rag_agent.py": "Agent RAG principal",
        "agent_5_validation/mcp_tools/mcp_server.py": "Serveur MCP Agent 5",
        "agent_5_validation/validation/hybrid_validator.py": "Validateur hybride",
        "agent_5_validation/validation/semantic_validator.py": "Validateur sémantique",
        "agent_5_validation/validation/llm_judge.py": "LLM Judge",
        "agent_5_validation/execution/sandbox_executor.py": "Exécuteur sandbox",
        "agent_5_validation/execution/vm_executor.py": "Exécuteur VM",
        "agent_5_validation/self_correction/corrector.py": "Agent de correction",
    }
    
    missing_files = []
    
    for file_path, description in required_files.items():
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {description}: {file_path}")
        else:
            print(f"  ❌ MANQUANT - {description}: {file_path}")
            missing_files.append(file_path)
    
    print()
    
    if missing_files:
        print("⚠️  Fichiers manquants détectés:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nAssurez-vous que tous les fichiers sources sont présents.")
    else:
        print("✅ Tous les fichiers requis sont présents!")
    
    print()


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║       CONFIGURATION DU PROJET NMAP-AI                     ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Vérifier la structure
    check_project_structure()
    
    # Créer les __init__.py
    create_init_files()
    
    # Créer les configs
    create_config_files()
    
    # Instructions pour les dépendances
    install_dependencies()
    
    print("="*60)
    print("🎯 Configuration terminée!")
    print("="*60)
    print()
    print("Prochaines étapes:")
    print("1. Installer les dépendances (voir commandes ci-dessus)")
    print("2. Démarrer les services dans cet ordre:")
    print("   a) Agent 1 - Complexity API")
    print("   b) Agent 5 - MCP Server")
    print("   c) Agent 2 - RAG Client")
    print("   d) Agent 3 - Diffusion Client")
    print()


if __name__ == "__main__":
    main()