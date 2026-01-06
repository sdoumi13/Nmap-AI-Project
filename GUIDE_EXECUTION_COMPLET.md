# GUIDE D'EXÉCUTION COMPLET - NMAP AI PROJECT

## 🎯 FLUX COMPLET DE L'ARCHITECTURE

```
USER QUERY (Prompt)
         ↓
    [ROUTER AGENT 1] ← décision centrale
         ├─ ÉTAPE 1: COMPREHENSION CHECK
         │   └─ La requête porte-t-elle sur nmap? ✓/✗
         │
         ├─ ÉTAPE 2: COMPLEXITY CLASSIFICATION (REST API)
         │   ├─ Analyse des keywords dans la requête
         │   ├─ Résultat: EASY / MEDIUM / HARD
         │   └─ Choix d'agent: RAG (Easy) ou DIFFUSION (Medium/Hard)
         │
         ├─ ÉTAPE 3: COMMAND GENERATION
         │   ├─ Si RAG: génère avec exemples de ChromaDB
         │   └─ Si DIFFUSION: génère avec modèle diffusion T5
         │
         └─ ÉTAPE 4: APPEL MCP CLIENT → Agent 5 (REST API)
              ↓
         [AGENT 5 MCP SERVER] (Serveur central)
              ├─ VALIDATION: Hybrid Semantic + LLM Judge
              ├─ AUTO-CORRECTION: Boucle de correction si invalide
              ├─ SANDBOX TEST: Docker container local
              ├─ VM EXECUTION: SSH vers Ubuntu VM (192.168.188.128)
              ├─ SELF-CORRECTION: Vérifie les résultats
              └─ RAPPORT STRUCTURÉ
                   ↓
         RÉPONSE UTILISATEUR
```

---

## 📋 PRÉREQUIS ET SERVICES À LANCER

### Service 1: Agent 5 MCP Server (DOIT ÊTRE LANCÉ EN PREMIER)
```bash
# Terminal 1
cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
.\venv\Scripts\python.exe agent_5_validation\run_agent5.py
```

**Attendu:**
```
🚀 Starting Agent 5 MCP Server...
✅ Agent 5 MCP Server ready on http://0.0.0.0:5000
```

**Composants lancés:**
- `MCPServer` (validation, correction, sandbox, VM execution)
- `HybridValidator` (sémantique + LLM Judge)
- `SandboxExecutor` (Docker)
- `VMExecutor` (SSH vers Ubuntu)
- `AutoCorrector` (correction auto des commandes invalides)

---

### Service 2: Complexity Classification API (REST API)
```bash
# Terminal 2
cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
.\venv\Scripts\python.exe -m uvicorn agent_1_router.complexity:app --port 7000 --log-level info
```

**Attendu:**
```
╔════════════════════════════════════════════╗
║   NMAP COMPLEXITY CLASSIFIER API          ║
║   Easy → RAG | Medium/Hard → Diffusion    ║
╚════════════════════════════════════════════╝

INFO:     Uvicorn running on http://127.0.0.1:7000
```

**Endpoints:**
- `POST /classify` - Classifie la complexité de la requête
- `GET /docs` - Swagger documentation

---

### Service 3: Ollama (pour RAG et Diffusion)
```bash
# Terminal 3 - Garder en arrière-plan
ollama run llama3
```

**Attendu:**
```
pulling manifest ⠋
success
>>> 
```

> **NOTE:** Ollama doit rester actif pendant toute la session. Vous pouvez minimiser cette fenêtre.

---

## 🚀 LANCER LE ROUTER AGENT

```bash
# Terminal 4 - Principal
cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
.\venv\Scripts\python.exe run_router_main.py
```

**Attendu:**
```
======================================================================
      NMAP-AI CORRECTED ARCHITECTURE
      Query → Router (Complexity) → Agent → MCP (Validation) → Sandbox → VM
======================================================================

[*] Initializing RouterAgent...
🧠 Initialisation Comprehension Agent...
   📂 Chemin corpus: C:\Users\Public\M2\Deep Leaning\Nmap-AI-Project\datasets\rag_corpus_detailed.json
   📚 Chargement de 15 docs et 3 exemples...
   Chargement du modèle d'embeddings...
   ✅ Comprehension Agent prêt!
[*] Initializing ComplexityAgent...
[✓] ComplexityAgent ready
✓ RouterAgent ready
----------------------------------------------------------------------
Default Target: 192.168.188.128 (Ubuntu VM)
Complexity API: http://localhost:7000/classify
MCP Agent 5: http://localhost:5000/mcp/execute
----------------------------------------------------------------------

ROUTER >
```

---

## 💬 ENTRER DES REQUÊTES

### Exemple 1: Requête FACILE (Easy) → RAG

```
ROUTER > scan port 80 on target
Target (Default 192.168.188.128): 
Processing...
```

**Flux décrit:**
```
[STEP 1/4] COMPREHENSION CHECK
  ✅ VALID. Score: 0.95

[STEP 2/4] COMPLEXITY CLASSIFICATION
  Level: Easy
  Confidence: 0.85
  Recommended Agent: RAG
  Reasoning: Query contains basic scanning keywords: 2 matches

[STEP 3/4] COMMAND GENERATION (RAG)
  Generated: nmap -p 80 TARGET
  (Cherche des exemples similaires dans ChromaDB et génère avec llama3)

[STEP 4/4] MCP EXECUTION (AGENT 5)
  Validation: hybrid semantic + LLM judge
  Auto-correction: (si invalide, corrige automatiquement)
  Sandbox test: execute dans Docker container
  VM execution: SSH vers 192.168.188.128
  Final Status: success ✓
```

---

### Exemple 2: Requête MOYENNE (Medium) → DIFFUSION

```
ROUTER > stealth scan with timing options
Target (Default 192.168.188.128): 
Processing...
```

**Flux décrit:**
```
[STEP 1/4] COMPREHENSION CHECK
  ✅ VALID. Score: 0.92

[STEP 2/4] COMPLEXITY CLASSIFICATION
  Level: Medium
  Confidence: 0.80
  Recommended Agent: DIFFUSION
  Reasoning: Query contains medium keywords: 2 matches

[STEP 3/4] COMMAND GENERATION (DIFFUSION)
  Generated: nmap -sS -T3 TARGET
  (Modèle diffusion T5 génère avec 15 steps de sampling)

[STEP 4/4] MCP EXECUTION (AGENT 5)
  Validation: hybrid semantic + LLM judge
  Auto-correction: (si invalide, corrige automatiquement)
  Sandbox test: execute dans Docker container
  VM execution: SSH vers 192.168.188.128
  Final Status: success ✓
```

---

### Exemple 3: Requête DIFFICILE (Hard) → DIFFUSION

```
ROUTER > comprehensive network reconnaissance with os detection
Target (Default 192.168.188.128): 
Processing...
```

**Flux décrit:**
```
[STEP 1/4] COMPREHENSION CHECK
  ✅ VALID. Score: 0.90

[STEP 2/4] COMPLEXITY CLASSIFICATION
  Level: Hard
  Confidence: 0.85
  Recommended Agent: DIFFUSION
  Reasoning: Query contains advanced keywords: 2 matches

[STEP 3/4] COMMAND GENERATION (DIFFUSION)
  Generated: nmap -A -O -Pn TARGET
  (Modèle diffusion T5 pour commandes complexes)

[STEP 4/4] MCP EXECUTION (AGENT 5)
  Validation: hybrid semantic + LLM judge
  Auto-correction: (si invalide, corrige automatiquement)
  Sandbox test: execute dans Docker container
  VM execution: SSH vers 192.168.188.128
  Final Status: success ✓
```

---

## 🔄 DÉTAIL DU FLUX AGENT 5 (MCP SERVER)

Quand le Router envoie une commande à Agent 5:

### Phase 1: VALIDATION
```python
# Appel MCP Client du Router:
mcp_result = await mcp_client.execute_command(
    command="nmap -p 80 TARGET",
    intent="scan port 80 on target",
    target="192.168.188.128",
    agent_name="rag"  # ou "diffusion"
)

# Serveur MCP Agent 5 reçoit et valide:
HybridValidator.validate(command, intent)
  ├─ SemanticValidator: analyse sémantique
  └─ LLMJudge: demande à Claude si valide
```

**Résultat:**
- ✅ VALID → Continuer au sandbox
- ❌ INVALID → Passer à Auto-Correction

---

### Phase 2: AUTO-CORRECTION (si invalide)
```python
Corrector.correct_command(command, validation_errors)
  ├─ Analyse les erreurs
  ├─ Détermine les parties invalides
  └─ Génère version corrigée

Max 3 tentatives de correction
```

**Résultat:**
- ✅ Corrigée → Continuer au sandbox avec cmd corrigée
- ❌ Impossible à corriger → Retourner erreur

---

### Phase 3: SANDBOX TEST (Docker)
```python
SandboxExecutor.execute(command, target)
  ├─ Lance container Docker Ubuntu
  ├─ Execute la commande nmap
  ├─ Récupère output + exit code
  └─ Valide que la commande fonctionne
```

**Résultat:**
- ✅ Réussi → Continuer à VM Execution
- ❌ Échoué → Auto-correction (jusqu'à 3 fois)

---

### Phase 4: VM EXECUTION (SSH)
```python
VMExecutor.execute_on_vm(
    command="nmap -p 80 TARGET",
    target="192.168.188.128",
    username="user",
    password="password"
)
  ├─ Connexion SSH vers Ubuntu VM
  ├─ Execute la commande réelle
  ├─ Récupère output
  └─ Retourne résultats
```

**Résultat:**
```
Nmap scan report for 192.168.188.128
Host is up (0.0012s latency)

PORT   STATE SERVICE
80/tcp open  http

Nmap done at ...
```

---

### Phase 5: SELF-CORRECTION (Vérification)
```python
SelfCorrector.verify(
    original_command,
    command_executed,
    output,
    expected_behavior
)
  ├─ Vérifie les résultats
  ├─ Détecte anomalies
  └─ Suggère ajustements si nécessaire
```

---

### Phase 6: RAPPORT FINAL
```json
{
  "final_status": "success",
  "command": "nmap -p 80 192.168.188.128",
  "stages": {
    "comprehension": {
      "valid": true,
      "score": 0.95
    },
    "complexity": {
      "level": "Easy",
      "agent": "RAG"
    },
    "generation": {
      "success": true,
      "output": "nmap -p 80 TARGET"
    },
    "validation": {
      "valid": true,
      "method": "hybrid_semantic_llm"
    },
    "correction": {
      "needed": false
    },
    "sandbox": {
      "success": true,
      "exit_code": 0,
      "output": "Nmap scan report for 192.168.188.128..."
    },
    "vm_execution": {
      "success": true,
      "duration": "2.34s",
      "output": "HOST IS UP...",
      "open_ports": [80]
    },
    "self_correction": {
      "verified": true,
      "anomalies": []
    }
  },
  "timestamp": "2026-01-06T14:35:22.123456"
}
```

---

## 🛠️ COMMANDES UTILES

### Vérifier les services actifs
```bash
# Vérifier Router
curl http://localhost:7000/docs

# Vérifier MCP Agent 5
curl http://localhost:5000/docs

# Test de classification
curl -X POST http://localhost:7000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "scan port 80", "user_id": "test"}'
```

### Résultats d'exécution
Tous les rapports sont sauvegardés dans:
```
agent_5_validation/execution/reports/
```

### Logs
```bash
# Agent 5
tail -f agent_5_validation/logs/agent5.log

# Router
tail -f logs/router.log
```

---

## ❌ PROBLÈMES COURANTS

### "Connection refused on port 5000"
```bash
# Agent 5 n'est pas lancé
# Lancer: python agent_5_validation/run_agent5.py
```

### "Connection refused on port 7000"
```bash
# Complexity API n'est pas lancée
# Lancer: python -m uvicorn agent_1_router.complexity:app --port 7000
```

### "Ollama not responding"
```bash
# Ollama n'est pas en arrière-plan
# Lancer: ollama run llama3
```

### "Query rejected - not relevant to nmap"
```bash
# La requête ne porte pas sur nmap
# Exemple valide: "scan port 80"
# Exemple invalide: "what is the weather"
```

---

## 📊 RÉSUMÉ DU FLUX

| Étape | Composant | Entrée | Sortie | Port |
|-------|-----------|--------|--------|------|
| 1 | Router Agent | Requête utilisateur | Analyse de compréhension | - |
| 2 | Complexity API | Requête | Niveau + agent choisi | 7000 |
| 3a | RAG Agent | Requête | Commande nmap | - |
| 3b | Diffusion Model | Requête | Commande nmap | - |
| 4 | MCP Client (Router) | Commande | Appel vers Agent 5 | - |
| 5 | Agent 5 MCP Server | Commande | Validation + Correction + Sandbox + VM | 5000 |
| 6 | Hybrid Validator | Commande | Score validité | - |
| 7 | Sandbox Docker | Commande | Test local | - |
| 8 | VM Executor (SSH) | Commande | Résultats réels | SSH |
| 9 | Self-Corrector | Résultats | Vérification finale | - |

---

## ✅ CHECKLIST D'EXÉCUTION

```bash
□ Terminal 1: Agent 5 MCP Server lancé (port 5000)
□ Terminal 2: Complexity API lancée (port 7000)
□ Terminal 3: Ollama lancé (llama3 model)
□ Terminal 4: Router Agent lancé
□ Vérifier connectivité VM (192.168.188.128)
□ Entrer première requête: "scan port 80"
□ Vérifier succès complet dans tous les étapes
```

---

## 🎓 EXEMPLE COMPLET D'EXÉCUTION

```
ROUTER > scan port 80
Target (Default 192.168.188.128): 

======================================================================
[STEP 1/4] COMPREHENSION CHECK
  ✅ VALID. Score: 0.95

[STEP 2/4] COMPLEXITY CLASSIFICATION
  Level: Easy
  Confidence: 0.85
  Recommended Agent: RAG
  Reasoning: Query contains basic scanning keywords: 1 matches

[STEP 3/4] COMMAND GENERATION (RAG)
  Generated: nmap -p 80 TARGET

[STEP 4/4] MCP EXECUTION (AGENT 5)
  [Validation] ✓ Valid (score: 95/100)
  [Auto-Correction] - Not needed
  [Sandbox Test] ✓ Success (exit code: 0)
  [VM Execution] ✓ Success (2.34s)
  [Self-Correction] ✓ Verified

Final Status: SUCCESS ✓

Report:
{
  "final_status": "success",
  "command": "nmap -p 80 192.168.188.128",
  "stages": {
    "validation": {"valid": true, "score": 95},
    "sandbox": {"success": true},
    "vm_execution": {
      "success": true,
      "output": "PORT   STATE SERVICE\n80/tcp open  http"
    }
  },
  "timestamp": "2026-01-06T14:35:22.123456"
}

ROUTER >
```

---

## 📖 FICHIERS CLÉS

- [agent_1_router/run_router.py](agent_1_router/run_router.py) - Router principal
- [agent_1_router/complexity.py](agent_1_router/complexity.py) - Classifier API
- [agent_1_router/comprehension.py](agent_1_router/comprehension.py) - Compréhension
- [RAG/agent/rag_agent.py](RAG/agent/rag_agent.py) - Générateur RAG
- [diffusion_models/discrete_diffusion_nmap.py](diffusion_models/discrete_diffusion_nmap.py) - Générateur Diffusion
- [agent_5_validation/mcp_tools/mcp_server.py](agent_5_validation/mcp_tools/mcp_server.py) - Serveur MCP Agent 5
- [agent_5_validation/validation/hybrid_validator.py](agent_5_validation/validation/hybrid_validator.py) - Validateur hybride
- [agent_5_validation/execution/sandbox_executor.py](agent_5_validation/execution/sandbox_executor.py) - Sandbox Docker
- [agent_5_validation/execution/vm_executor.py](agent_5_validation/execution/vm_executor.py) - VM Executor SSH

---

**Créé:** 6 janvier 2026  
**Architecture:** Nmap AI Project - Corrected  
**Version:** 1.0 Complet
