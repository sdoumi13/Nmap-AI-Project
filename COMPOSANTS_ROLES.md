# 🎯 RÔLES DES COMPOSANTS - MCP Client, RAG, Diffusion

## 📊 SCHÉMA DE COMMUNICATION

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│                    "scan port 80 on target"                 │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   [ROUTER AGENT 1]                          │
│  1. Comprehension Check (valid nmap query?)                 │
│  2. Complexity Classification (EASY/MEDIUM/HARD)            │
│  ├─→ Si EASY: Utilise RAG                                   │
│  └─→ Si MEDIUM/HARD: Utilise DIFFUSION                      │
└────────────────────────────┬────────────────────────────────┘
                             ↓
        ┌────────────────────┴────────────────────┐
        ↓                                         ↓
   [RAG AGENT]                          [DIFFUSION AGENT]
   Récupère examples                    Génère avec modèle
   de ChromaDB +                        T5 diffusion
   LLM Ollama llama3                    + LLM Ollama
   Génère: nmap cmd                     Génère: nmap cmd
        ↓                                         ↓
        └────────────────────┬────────────────────┘
                             ↓
        ┌─────────────────────────────────────────┐
        │    [MCP CLIENT] (dans Router)           │
        │ Envoie la commande générée via HTTP     │
        │ POST à http://localhost:5000/mcp/execute│
        └──────────────┬──────────────────────────┘
                       ↓
        ┌──────────────────────────────────────────┐
        │  [AGENT 5 MCP SERVER] (validation-vm)   │
        │                                          │
        │  Phase 1: VALIDATION (hybrid)           │
        │  Phase 2: AUTO-CORRECTION               │
        │  Phase 3: SANDBOX TEST (Docker)         │
        │  Phase 4: VM EXECUTION (SSH)            │
        │  Phase 5: SELF-CORRECTION               │
        │  Phase 6: RAPPORT FINAL                 │
        └─────────────┬────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────────┐
        │       RÉSULTATS AU USER                 │
        │    Via ROUTER AGENT prompt              │
        └─────────────────────────────────────────┘
```

---

## 🎯 COMPOSANT 1: MCP CLIENT

### Qu'est-ce que c'est?
**MCP Client** = Client HTTP qui communique avec Agent 5 MCP Server

### Où est-il?
```
agent_1_router/run_router.py
   └─ Class: MCPClient
      └─ Method: execute_command()
```

### Quel est son rôle?
1. ✅ Prend la commande générée (RAG ou Diffusion)
2. ✅ L'envoie au MCP Server (Agent 5) via REST API
3. ✅ Reçoit la réponse avec validation + sandbox + VM results
4. ✅ Retourne le rapport final au Router

### Exemple d'appel:
```python
mcp_result = await mcp_client.execute_command(
    command="nmap -p 80 TARGET",
    intent="scan port 80",
    target="192.168.188.128",
    agent_name="rag"
)

# Résultat:
{
    "final_status": "success",
    "stages": {
        "validation": {...},
        "sandbox": {...},
        "vm_execution": {...}
    }
}
```

### Doit-il être lancé?
❌ **NON** - C'est une classe Python utilisée en interne par le Router  
Il n'a pas de serveur indépendant.

---

## 🎯 COMPOSANT 2: RAG AGENT

### Qu'est-ce que c'est?
**RAG** = Retrieval-Augmented Generation  
Génère des commandes nmap en cherchant des exemples similaires dans ChromaDB

### Où est-il?
```
RAG/agent/rag_agent.py
   └─ Class: NmapRagAgent
      └─ Method: process()
```

### Quel est son rôle?
1. ✅ Reçoit la requête utilisateur
2. ✅ Cherche les exemples similaires dans ChromaDB
3. ✅ Utilise Ollama llama3 LLM pour générer la commande
4. ✅ Retourne une commande nmap valide

### Utilisé pour:
- **EASY queries** (requêtes simples)
- Exemples: "scan port 80", "check services", "list open ports"

### Exemple:
```python
rag = NmapRagAgent()
result = rag.process({
    "user_query": "scan port 80 on target",
    "extracted_ip": "192.168.188.128"
})

# Résultat: "nmap -p 80 TARGET"
```

### Doit-il être lancé?
❌ **NON** - C'est appelé en interne par le Router  
Pas de serveur indépendant.

### Dépendances:
- ✅ ChromaDB (base de données vecteurs locales)
- ✅ Ollama llama3 (LLM local)

---

## 🎯 COMPOSANT 3: DIFFUSION MODEL

### Qu'est-ce que c'est?
**Diffusion** = Modèle T5 fine-tuned sur nmap commands  
Génère des commandes complexes via process diffusion

### Où est-il?
```
diffusion_models/discrete_diffusion_nmap.py
   └─ Class: DiscreteNmapDiffusionModel
      └─ Method: generate()
```

### Quel est son rôle?
1. ✅ Reçoit la requête utilisateur
2. ✅ Utilise un modèle T5 diffusion (15 étapes de sampling)
3. ✅ Génère des commandes nmap complexes/précises
4. ✅ Retourne une commande nmap générée

### Utilisé pour:
- **MEDIUM queries** (requêtes modérées)
- **HARD queries** (requêtes complexes)
- Exemples: "stealth scan", "OS detection", "comprehensive scan"

### Exemple:
```python
diffusion = DiscreteNmapDiffusionModel()
result = diffusion.generate(
    query="stealth scan on port 443",
    target="192.168.188.128"
)

# Résultat: "nmap -sS -p 443 -T3 TARGET"
```

### Doit-il être lancé?
❌ **NON** - C'est appelé en interne par le Router  
Pas de serveur indépendant.

### Dépendances:
- ✅ Modèle T5 téléchargé (dans diffusion_models/nmap_diffusion_checkpoint/)
- ✅ Ollama llama3 (optionnel pour refinement)

---

## 🔄 FLUX COMPLET

```
1. USER TAPE: "scan port 80"
   ↓
2. ROUTER reçoit la requête
   ├─ Complexity Classification → "EASY"
   └─ Décision: Utiliser RAG
   ↓
3. RAG AGENT
   ├─ Cherche examples dans ChromaDB
   ├─ Génère avec Ollama llama3
   └─ Retourne: "nmap -p 80 TARGET"
   ↓
4. MCP CLIENT (dans Router)
   ├─ Prend la commande
   └─ Envoie à http://localhost:5000/mcp/execute
   ↓
5. AGENT 5 MCP SERVER
   ├─ Phase 1: VALIDATION ✅
   ├─ Phase 2: AUTO-CORRECTION (si besoin)
   ├─ Phase 3: SANDBOX TEST ✅
   ├─ Phase 4: VM EXECUTION ✅
   └─ Retourne rapport complet
   ↓
6. ROUTER reçoit le rapport
   └─ Affiche les résultats au user
```

---

## 📋 SERVICES À LANCER

### Service 1: Agent 5 MCP SERVER (REQUIS)
```bash
# Terminal 1
cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
python agent_5_validation\run_agent5.py
```

**Status:** ✅ Attend les commandes du Router via REST API

### Service 2: Complexity Classifier API (REQUIS)
```bash
# Terminal 2
cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
python -m uvicorn agent_1_router.complexity:app --port 7000
```

**Status:** ✅ Classe les requêtes EASY/MEDIUM/HARD

### Service 3: Ollama (REQUIS)
```bash
# Terminal 3
ollama run llama3
```

**Status:** ✅ LLM utilisé par RAG et Diffusion

### Service 4: Router Agent (PRINCIPAL)
```bash
# Terminal 4
cd c:\Users\Public\M2\Deep Leaning\Nmap-AI-Project
python run_router_main.py
```

**Status:** ✅ Interface utilisateur - c'est là qu'on entre les requêtes

---

## ❌ COMPOSANTS QUI NE DOIVENT PAS ÊTRE LANCÉS INDÉPENDAMMENT

| Composant | Pourquoi |
|-----------|---------|
| **RAG Agent** | ❌ Appelé par Router, pas de serveur |
| **Diffusion Model** | ❌ Appelé par Router, pas de serveur |
| **MCP Client** | ❌ Classe utilisée par Router, pas de serveur |
| **Hybrid Validator** | ❌ Utilisé par Agent 5, pas de serveur |
| **Sandbox Executor** | ❌ Utilisé par Agent 5, pas de serveur |
| **VM Executor** | ❌ Utilisé par Agent 5, pas de serveur |

---

## ✅ COMPOSANTS À LANCER (4 SERVICES)

```
1. python agent_5_validation\run_agent5.py         (Agent 5 MCP Server)
2. python -m uvicorn agent_1_router.complexity:app  (Complexity API)
3. ollama run llama3                               (LLM Ollama)
4. python run_router_main.py                       (Router - interface user)
```

---

## 🎯 RÉSUMÉ RAPIDE

| Composant | Type | Lance? | Rôle |
|-----------|------|--------|------|
| **Router** | Service | ✅ OUI | Interface user + orchestration |
| **Agent 5 MCP Server** | Service | ✅ OUI | Validation + Sandbox + VM |
| **Complexity API** | Service | ✅ OUI | Classification EASY/MEDIUM/HARD |
| **Ollama llama3** | Service | ✅ OUI | LLM |
| **RAG Agent** | Module | ❌ NON | Génère commands (EASY queries) |
| **Diffusion Model** | Module | ❌ NON | Génère commands (MEDIUM/HARD queries) |
| **MCP Client** | Module | ❌ NON | Communique Router → Agent5 |
| **Validators** | Module | ❌ NON | Valide les commandes (dans Agent5) |
| **Executors** | Module | ❌ NON | Exécute les commandes (dans Agent5) |

---

## 📊 ARCHITECTURE RÉELLE

```
VOUS LANCEZ:
┌──────────────────────────────────────┐
│  4 SERVICES (4 Terminaux)            │
├──────────────────────────────────────┤
│ 1. Agent 5 MCP Server                │
│ 2. Complexity Classifier API          │
│ 3. Ollama llama3 LLM                 │
│ 4. Router Agent (interface user)     │
└──────────────────────────────────────┘

À L'INTÉRIEUR:
┌──────────────────────────────────────┐
│  Router utilise en interne:          │
├──────────────────────────────────────┤
│ - RAG Agent (si EASY)                │
│ - Diffusion Model (si MEDIUM/HARD)  │
│ - MCP Client (pour envoyer)          │
│ - Comprehension Agent                │
│ - Complexity Agent                   │
└──────────────────────────────────────┘

À L'INTÉRIEUR D'AGENT 5:
┌──────────────────────────────────────┐
│  Agent 5 utilise en interne:         │
├──────────────────────────────────────┤
│ - Hybrid Validator                   │
│ - Self-Corrector                     │
│ - Sandbox Executor (Docker)          │
│ - VM Executor (SSH)                  │
│ - Self-Correction Agent              │
└──────────────────────────────────────┘
```

---

## 🎓 EXEMPLE COMPLET

```
USER: "scan port 80"
  ↓
ROUTER reçoit + analyse
  ├─ Complexity Classification API → "EASY"
  ├─ Décision: RAG
  ↓
RAG AGENT
  ├─ ChromaDB: Cherche exemples similaires
  ├─ Ollama llama3: Génère
  └─ Retourne: "nmap -p 80 TARGET"
  ↓
MCP CLIENT (dans Router)
  └─ Envoie à Agent 5: POST /mcp/execute
  ↓
AGENT 5 MCP SERVER
  ├─ Hybrid Validator: Valide ✅
  ├─ Sandbox Executor: Teste dans Docker ✅
  ├─ VM Executor: Exécute sur VM ✅
  └─ Retourne: Rapport complet
  ↓
ROUTER affiche résultats
```

---

## ❓ FAQ

**Q: Le RAG et Diffusion doivent-ils être lancés?**  
A: Non, ils sont appelés par le Router en interne

**Q: Combien de services dois-je lancer?**  
A: 4 services (Router, Agent5, Complexity API, Ollama)

**Q: Quel est le rôle du MCP Client?**  
A: Envoyer la commande du Router au serveur Agent5 via REST API

**Q: Quel est la différence RAG vs Diffusion?**  
A: RAG = retrieval+generation (EASY), Diffusion = complex generation (MEDIUM/HARD)

**Q: Agent 5 lance seul ou via Router?**  
A: Agent 5 se lance dans un Terminal, puis Router l'appelle via REST API

---

**Créé:** 6 janvier 2026  
**Status:** ✅ Complet et expliqué
