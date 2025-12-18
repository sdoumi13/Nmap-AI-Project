# Nmap-AI-Project

"""
NMAP-AI Agent 5 - Complete Implementation
Validation + Self-Correction + MCP + VM Execution

Project Structure:
agent5/
├── __init__.py
├── validation/
│   ├── __init__.py
│   ├── semantic_validator.py      # Step 1: Semantic rules
│   ├── llm_judge.py                # Step 2: LLM fallback
│   └── hybrid_validator.py         # Step 3: Combined validation
├── mcp/
│   ├── __init__.py
│   ├── protocol.py                 # Step 4: MCP message schemas
│   ├── server.py                   # Step 5: MCP server
│   └── client.py                   # Step 6: MCP client
├── execution/
│   ├── __init__.py
│   ├── sandbox_executor.py         # Step 7: Docker sandbox
│   └── vm_executor.py              # Step 8: VM execution
├── self_correction/
│   ├── __init__.py
│   └── corrector.py                # Step 9: Self-correction loop
└── orchestrator.py                 # Step 10: Main orchestrator
"""


# Nmap AI Project - Agent 1 Router

## Architecture

Ce projet implémente un système intelligent de routage pour la génération de commandes Nmap basé sur 3 corpus détaillés :

### 📁 Structure

```
Nmap-AI-Project/
├── agent_1_router/
│   ├── __init__.py
│   ├── comprehension.py      # TF-IDF & SBERT Filter (utilise RAG corpus)
│   ├── complexity.py          # Classification Easy/Medium/Hard (utilise Fine-tuning corpus)
│   └── run_router.py          # Point d'entrée principal
├── datasets/
│   ├── rag_corpus_detailed.json           # Corpus pour RAG (15 entrées enrichies)
│   ├── finetuning_corpus_detailed.json    # Corpus pour Fine-tuning (5 conversations)
│   └── diffusion_corpus_detailed.json     # Corpus pour Diffusion (20 paires)
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le router
python agent_1_router/run_router.py
```

## 📊 Corpus Enrichis

### 1. RAG Corpus (rag_corpus_detailed.json)
- **15 entrées** avec contexte, explications, prérequis, temps d'exécution
- Utilisé par `ComprehensionAgent` pour la détection de pertinence
- Métadonnées : catégories, difficulté, tags sémantiques

### 2. Fine-tuning Corpus (finetuning_corpus_detailed.json)
- **5 conversations multi-turn** avec raisonnement expert
- Utilisé par `ComplexityAgent` pour identifier les patterns de complexité
- Niveaux : easy → medium → hard

### 3. Diffusion Corpus (diffusion_corpus_detailed.json)
- **20 paires description→commande** avec contexte d'embedding
- Prêt pour entraînement de modèles seq2seq
- Niveaux de complexité 1-10

## 🎯 Fonctionnement

1. **Comprehension Agent** : Analyse la requête avec TF-IDF + SBERT sur le corpus RAG
2. **Complexity Agent** : Classifie en Easy/Medium/Hard selon les patterns détectés
3. **Routing** : Dirige vers l'agent approprié selon la complexité

## 💡 Exemples

```bash
# Easy Query
> scan port 80
✅ PERTINENT | ROUTAGE VERS: Agent Easy (KG-RAG)

# Medium Query  
> aggressive scan with os detection
✅ PERTINENT | ROUTAGE VERS: Agent Medium (Fine-Tuned Model)

# Hard Query
> stealth scan ipv6 with nse scripts bypassing firewall
✅ PERTINENT | ROUTAGE VERS: Agent Hard (Diffusion/ReAct)
```

## 📈 Statistiques de Session

Le router affiche à la sortie :
- Nombre total de requêtes
- Pertinentes vs Irrelevantes
- Distribution Easy/Medium/Hard

## 🔧 Configuration

Pour personnaliser les chemins des corpus, modifiez les paramètres dans `run_router.py` :

```python
comp_agent = ComprehensionAgent(corpus_path='path/to/your/rag_corpus.json')
complex_agent = ComplexityAgent(finetuning_corpus_path='path/to/your/ft_corpus.json')
