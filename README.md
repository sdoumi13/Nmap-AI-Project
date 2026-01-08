# Nmap-AI-Project

"""


# Nmap AI Project - Agent 1 Router

## Architecture

Ce projet implémente un système intelligent de routage pour la génération de commandes Nmap basé sur 3 corpus détaillés :


```

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le router
python agent_1_router/run_router.py
```

##  Corpus Enrichis

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

## Fonctionnement

1. **Comprehension Agent** : Analyse la requête avec TF-IDF + SBERT sur le corpus RAG
2. **Complexity Agent** : Classifie en Easy/Medium/Hard selon les patterns détectés
3. **Routing** : Dirige vers l'agent approprié selon la complexité


