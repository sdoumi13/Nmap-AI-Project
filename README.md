# Nmap-AI-Project

"""
##

# Nmap AI Project - 


## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# chackend Lancement
python agent_1_router/run_router.py
python agent5_validation/run_agent5.py
python agent5_validation/mcp_tools/mcp_server.py   
```

##  Corpus

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


1. **Comprehension Agent** : Analyse la requête avec TF-IDF + SBERT sur le corpus RAG
2. **Complexity Agent** : Classifie en Easy/Medium/Hard selon les patterns détectés
3. **Routing** : Dirige vers l'agent approprié selon la complexité

## APP- Router Example 
![LLM](/Annexe/1.png)
![LLM](/Annexe/2.png)
