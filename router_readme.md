#  Agent 1 - Router Agent

> **Orchestrateur Intelligent du Système Multi-Agents Nmap-AI**  
> Analyse, classifie et route les requêtes vers l'agent approprié avec validation complète

---

##  Vue d'ensemble

L'**Agent 1 (Router)** est le cerveau central du système Nmap-AI. Il effectue quatre tâches critiques :

```
┌────────────────────────────────────────────────────────────┐
│                    AGENT 1 - ROUTER                        │
│                                                            │
│  User Query → Comprehension → Complexity → Route → MCP    │
└────────────────────────────────────────────────────────────┘
```

### Rôle Principal

| Étape | Fonction | Objectif |
|-------|----------|----------|
| **1. Comprehension** | Filtre les requêtes | Rejeter le bruit non-Nmap |
| **2. Complexity** | Classifie la difficulté | Easy / Medium / Hard |
| **3. Routing** | Sélectionne l'agent | RAG (simple) ou Diffusion (complexe) |
| **4. Execution** | Envoie à Agent 5 | Validation + Correction + Sandbox + VM |

---

##  Architecture

### Structure des Fichiers

```
agent_1_router/
├── __init__.py
├── comprehension.py         
├── complexity.py             
├── distributed_routing.py    
└── run_router.py            
```

---

## Pipeline de Traitement

### Vue d'ensemble du Workflow

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: COMPREHENSION CHECK                                 │
│ ────────────────────────────────────────────────────────    │
│ TF-IDF (30%) + SBERT (30%) + SLM (40%)                     │
│                                                             │
│     Relevant (score ≥ 0.25) → Continue                      │
│ ❌ Irrelevant (score < 0.25) → REJECT                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: COMPLEXITY CLASSIFICATION                           │
│ ────────────────────────────────────────────────────────    │
│ Corpus Similarity (30%) + Keywords (25%) + SLM (45%)       │
│                                                             │
│ 🟢 Easy    → RAG Agent                                      │
│ 🟡 Medium  → Diffusion Agent                                │
│ 🔴 Hard    → Diffusion Agent                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: COMMAND GENERATION                                  │
│ ────────────────────────────────────────────────────────    │
│ IF Easy:    → DistributedRAGClient (192.168.1.218:8000)     │
│ IF Medium/Hard: → DiffusionClient or Fine-Tunnig Lora (192.168.1.169:9000)│
│                                                             │
│ Output: nmap command string                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: MCP EXECUTION PIPELINE                              │
│ ────────────────────────────────────────────────────────    │
│ 1. Semantic Validation   → Syntax rules                     │
│ 2. LLM Judge            → Mistral semantic check            │
│ 3. Self-Correction      → Fix errors                        │
│ 4. Sandbox Testing      → Docker safe exec                  │
│ 5. VM Execution         → Real target scan                  │
│                                                             │
│    Success → Return results                                 │
│ ❌ Failure → Return error report                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Modules Détaillés

### 1. `comprehension.py` 

**Objectif :** Déterminer si la requête est pertinente pour Nmap (filtre anti-bruit)

#### Architecture Multi-Couches

```
┌──────────────────────────────────────────────────────────┐
│              COMPREHENSION AGENT                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Layer 1: TF-IDF Vectorization (30%)                     │
│  ├─ Compare query vs Nmap corpus                         │
│  ├─ Compare query vs Noise corpus                        │
│  └─ Score: max_nmap - max_noise                          │
│                                                          │
│  Layer 2: SBERT Semantic Embedding (30%)                 │
│  ├─ Model: all-MiniLM-L6-v2                              │
│  ├─ Compute cosine similarity                            │
│  └─ Score vs reference Nmap embedding                    │
│                                                          │
│  Layer 3: SLM Deep Analysis (40%)                        │
│  ├─ Model: Qwen2.5-Coder-3B-Instruct                     │
│  ├─ API: http://192.168.11.1:1234                        │
│  └─ Returns: {is_nmap_related, confidence, reasoning}    │
│                                                          │
│  ┌────────────────────────────────────┐                  │
│  │ WEIGHTED VOTING                    │                  │
│  │ final_score = tfidf*0.3 +          │                  │
│  │               sbert*0.3 +          │                  │
│  │               slm*0.4              │                  │
│  └────────────────────────────────────┘                  │
│                                                          │
│  Decision: relevant = (final_score ≥ 0.25)              │
└──────────────────────────────────────────────────────────┘

```

#### Corpus Utilisé

- **Nmap Corpus** : `datasets/rag_corpus_detailed.json`
  - 150+ exemples de requêtes Nmap valides
  - Tags : intent, context, command, use_cases, related_concepts
  
- **Noise Corpus** : Hardcoded examples
  - Requêtes non-Nmap : météo, recettes, programmation générale, etc.

![Noise](/Annexe//image2.png)

#### Exemple de Traitement

**Input :** `"Lance un scan stealth et utilise le fragment de paquets pour evade le firewall"`

**Output :**
```python
{
  "relevant": True,
  "score": 0.699,
  "reason": "The query mentions stealth techniques and fragmenting packets...",
  "layers": {
    "tfidf_nmap": 0.221,
    "tfidf_noise": 0.000,
    "sbert_semantic": 0.343,
    "slm_relevant": True,
    "slm_confidence": 0.950
  },
  "extracted_intent": "Perform stealth SYN scan with packet fragmentation for firewall evasion"
}
```

**Visualisation (voir Annexe/image1.png) :**

```
[Comprehension] Analyzing: 'Lance un scan stealth et utilise le fragment...'
  [TF-IDF] Nmap: 0.221 | Noise: 0.000
  [SBERT] Semantic: 0.343
  [SLM] Related: True | Conf: 0.950
  [BOOST] Critical keyword detected (+0.15)
  [FINAL] Score: 0.699 | Threshold: 0.25 | Relevant: True

✅ VALID. Score: 0.70
```
![Comprehension](/Annexe//image1.png)

---

### 2. `complexity.py` 

**Objectif :** Classifier la complexité de la requête (Easy / Medium / Hard)

#### Architecture Hybride

```
┌──────────────────────────────────────────────────────────┐
│              COMPLEXITY CLASSIFIER                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Layer 1: Corpus-based Similarity (30%)                  │
│  ├─ Index corpus by difficulty:                          │
│  │   • easy_examples (complexity ≤ 3)                    │
│  │   • medium_examples (4 ≤ complexity ≤ 6)             │
│  │   • hard_examples (complexity > 6)                    │
│  ├─ Jaccard similarity with query                        │
│  └─ Best match scores per category                       │
│                                                          │
│  Layer 2: Keyword Heuristics (25%)                       │
│  ├─ Simple keywords: scan, ping, port, basic...          │
│  ├─ Complex keywords: stealth, evade, fragment, brute... │
│  └─ Returns: (level, confidence)                         │
│                                                          │
│  Layer 3: SLM Classification (45%)                       │
│  ├─ Few-shot prompt with corpus examples                 │
│  ├─ Model: Qwen2.5-Coder-3B-Instruct                     │
│  └─ Returns: {complexity, confidence, reasoning}         │
│                                                          │
│  ┌────────────────────────────────────┐                  │
│  │ WEIGHTED VOTING                    │                  │
│  │ votes[level] += corpus*0.30 +      │                  │
│  │                keywords*0.20 +     │                  │
│  │                slm*0.50            │                  │
│  └────────────────────────────────────┘                  │
│                                                          │
│  Final Level: argmax(votes)                              │
└──────────────────────────────────────────────────────────┘
```

#### Critères de Classification

| Niveau | Caractéristiques | Exemples de Commandes |
|--------|------------------|----------------------|
| **🟢 EASY** | • Scan basique<br>• 1-2 flags simples<br>• Pas d'évasion | `nmap -p- 192.168.1.0/24`<br>`nmap -sn 10.0.0.0/24` |
| **🟡 MEDIUM** | • Détection OS/Service<br>• 3-5 flags<br>• Scripts NSE standards | `nmap -sV -O -p 22,80,443 target`<br>`nmap --script default target` |
| **🔴 HARD** | • Techniques stealth<br>• Évasion IDS/Firewall<br>• NSE complexes<br>• Fragmentation, decoys | `nmap -sS -f -D RND:10 target`<br>`nmap --script brute target` |

#### Corpus Utilisés

1. **RAG Corpus** (`rag_corpus_detailed.json`)
   - Exemples avec field `difficulty: easy|medium|hard`

2. **Diffusion Corpus** (`diffusion_corpus_detailed.json`)
   - Exemples avec field `complexity_level: 1-10`
   - Mapping : ≤3=Easy, 4-6=Medium, >6=Hard

3. **Finetuning Corpus** (`finetuning_corpus_detailed.json`)
   - Conversations annotées avec `difficulty`


```

```

---

### 3. `distributed_routing.py` 

**Objectif :** Communiquer avec les agents distribués (RAG et Diffusion)

#### Architecture Distribuée

```
┌─────────────────────────────────────────────────────┐
│           DISTRIBUTED ROUTING                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌────────────────────────────────────┐             │
│  │  DistributedRAGClient              │             │
│  │  • URL: 192.168.1.218:8000         │             │
│  │  • Endpoint: /generate_command     │             │
│  │  • Timeout: 60s                    │             │
│  │  • Use: Easy queries (conf > 0.7)  │             │
│  └────────────────────────────────────┘             │
│                                                     │
│  ┌────────────────────────────────────┐             │
│  │  LocalDiffusionClient              │             │
│  │  • URL: 192.168.1.169:9000         │             │
│  │  • Endpoint: /generate             │             │
│  │  • Timeout: 90s                    │             │
│  │  • Use: Medium/Hard queries        │             │
│  └────────────────────────────────────┘             │
│                                                     │
│  ┌────────────────────────────────────┐             │
│  │  DistributedRouter                 │             │
│  │  • Smart routing logic              │             │
│  │  • Automatic fallback (RAG → Diff) │             │
│  │  • Health checks                    │             │
│  └────────────────────────────────────┘             │
└─────────────────────────────────────────────────────┘
```

### 4. `run_router.py` 🚀

**Objectif :** Orchestrateur principal exposant une API REST (FastAPI)

#### Architecture du Service

```
┌─────────────────────────────────────────────────────┐
│            ROUTER AGENT (FastAPI)                   │
│            Port: 8001                               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Endpoints:                                         │
│  ┌────────────────────────────────────┐             │
│  │ GET  /          → Service info     │             │
│  │ GET  /health    → Health check     │             │
│  │ POST /route     → Main pipeline    │             │
│  └────────────────────────────────────┘             │
│                                                     │
│  Components:                                        │
│  ┌────────────────────────────────────┐             │
│  │ ComprehensionAgent                 │             │
│  │ ComplexityAgent                    │             │
│  │ DistributedRouter                  │             │
│  │ MCPClient (Agent 5)                │             │
│  └────────────────────────────────────┘             │
└─────────────────────────────────────────────────────┘
```

#### Classe Principale


---

## 🚀 Installation & Configuration

### Prérequis Système

- **Python :** 3.8+
- **RAM :** 4GB minimum (8GB recommandé pour SLM)
- **Réseau :** Accès aux agents distribués

### 1. Installation des Dépendances

```bash
# Créer environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows

# Installer packages
pip install requirements.txt

### 2. Configuration des URLs

Éditer les URLs dans les fichiers selon votre réseau :

**`distributed_routing.py` :**
```python
COLLEAGUE_RAG_URL = "http://192.168.1.218:8000"  
DIFFUSION_LOCAL_URL = "http://192.168.1.169:9000"  
```

**`comprehension.py` & `complexity.py` :**
```python
SLM_API_URL = "http://192.168.11.1:1234/v1/chat/completions"  
```

**`run_router.py` :**
```python
MCP_AGENT5_URL = "http://localhost:5002" 
```

### 3. Vérification des Services

```bash
# Vérifier SLM (Qwen2.5)
curl http://192.168.11.1:1234/v1/models

# Vérifier RAG du collègue
curl http://192.168.1.218:8000/health

# Vérifier Diffusion local
curl http://192.168.1.169:9000/health

# Vérifier Agent 5 (MCP)
curl http://localhost:5002/health
```

### 4. Structure des Données

Assurez-vous que les corpus sont dans `datasets/` :

```
datasets/
├── rag_corpus_detailed.json          # Pour ComprehensionAgent
├── diffusion_corpus_detailed.json    # Pour ComplexityAgent
└── finetuning_corpus_detailed.json   # Pour ComplexityAgent
```

### 5. Lancement du Service

```bash
cd agent_1_router
python run_router.py

# Le service démarre sur http://0.0.0.0:8001
```

---


