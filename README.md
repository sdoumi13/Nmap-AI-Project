# Nmap-AI Project

## Team Members

- **Role 1:** DOUMI SALMA — Complexity Agent & Validation (MCP) Agent
- **Role 2:** — AFROUKH ABDELLAH \_ Nmap Discrete Diffusion Agent
- **Role 3:** —
- **Role 4:** —
- **Role 5:** —

---

## Project Overview

The **Nmap-AI Project** is an intelligent multi-agent system designed to understand, classify, generate, and safely execute **Nmap commands** using AI techniques (RAG,Fine-Tunnig, Diffusion models, and SLMs).

At the core of the system is **Agent 1 – Router Agent**, which acts as the brain of the architecture.

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# chackend Lancement
python agent_1_router/run_router.py
python agent5_validation/run_agent5.py
python agent5_validation/mcp_tools/mcp_server.py

```

# Agent 1 - Router Agent

> **Intelligent Orchestrator of the Nmap-AI Multi-Agent System**  
> Analyzes, classifies, and routes user queries to the appropriate agent with full validation.

---

## Global Workflow

User Query → Comprehension → Complexity → Routing → MCP Execution

### Rôle Principal

| Étape                | Fonction                | Objectif                               |
| -------------------- | ----------------------- | -------------------------------------- |
| **1. Comprehension** | Filtre les requêtes     | Rejeter le bruit non-Nmap              |
| **2. Complexity**    | Classifie la difficulté | Easy / Medium / Hard                   |
| **3. Routing**       | Sélectionne l'agent     | RAG (simple) /LoRA-fine-tuned T5-small-Phi-4 /Diffusion (complexe)   |
| **4. Execution**     | Envoie à Agent 5        | Validation + Correction + Sandbox + VM |

---

## Architecture

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
│ 🟡 Medium  → LoRA-fine-tuned T5-small / Phi-4 on                               │
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

## Modules Détaillés

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

![Noise](/Annexe//image3.png)

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

![Comprehension](/Annexe//image1.png)

#### Corpus Utilisés

1. **RAG Corpus** (`rag_corpus_detailed.json`)

   - Exemples avec field `difficulty: easy|medium|hard`

2. **Diffusion Corpus** (`diffusion_corpus_detailed.json`)

   - Exemples avec field `complexity_level: 1-10`
   - Mapping : ≤3=Easy, 4-6=Medium, >6=Hard

3. **Finetuning Corpus** (`finetuning_corpus_detailed.json`)
   - Conversations annotées avec `difficulty`

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

### 4. `run_router.py`

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

---

## Installation & Configuration

### Prérequis Système

- **Python :** 3.8+
- **RAM :** 4GB minimum (8GB recommandé pour SLM)
- **Réseau :** Accès aux agents distribués

### 1. Installation des Dépendances

````bash
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
````

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

## APP- Router Example

![LLM](/Annexe/1.png)
![LLM](/Annexe/2.png)

---

# Nmap Discrete Diffusion Agent

## Overview

AI-powered system that generates Nmap commands from natural language using **discrete diffusion** - iteratively refining commands from noisy states to valid Nmap syntax.

**Key Features**: T5-based (60M params), semantic-aware noise generation, FastAPI server, 4638 training commands

---

## Architecture

### Model: T5-Small + Discrete Diffusion

- **Base**: T5ForConditionalGeneration (6 encoder/decoder layers, 512 hidden dim)
- **Diffusion**: Progressive denoising: `nmap` → `+scan` → `+ports` → `+flags` → `complete`
- **Input**: `"refine: {noisy_cmd} | query: {nl_query}"`
- **Output**: Next refined command

### Workflow

```
Training:  Clean Command → Noise Generator → Training Pairs (noisy→clean) → T5 Training
Inference: NL Query → Intent Extraction → Iterative Refinement (max 15 steps) → Final Command
```

### Core Components

- **NmapNoiseGenerator**: Creates 7-level semantic noise sequences
- **NmapDiscreteDiffusionLM**: T5 wrapper with command validation
- **DiscreteDiffusionSampler**: Iterative refinement engine
- **DiffusionTrainer**: Training loop with checkpointing

---

## Discrete Diffusion Approach

### Forward Process: Semantic Noise Generation

Unlike standard diffusion that adds random Gaussian noise, we use **semantic degradation** that preserves command structure while progressively removing components.

**Noise Categories** (in semantic order):

1. **Scan Type** (`-sS`, `-sT`, `-sU`) - Core scanning method
2. **Port Specification** (`-p 80,443`, `-p-`, `-F`) - What to scan
3. **OS/Version Detection** (`-O`, `-sV`, `-A`) - Deep inspection
4. **Scripts** (`--script vuln`, `--script default`) - Advanced functionality
5. **Timing** (`-T0` to `-T5`) - Speed control
6. **Other Flags** (`-Pn`, `--traceroute`, `-D`) - Special options

**Example Noise Sequence**:

```
t=0: nmap
t=1: nmap -sS 192.168.1.0/24
t=2: nmap -sS -p 80,443 192.168.1.0/24
t=3: nmap -sS -p 80,443 -sV 192.168.1.0/24
t=4: nmap -sS -p 80,443 -sV --script vuln 192.168.1.0/24  [CLEAN]
```

**Key Insight**: Each step is a valid (though incomplete) Nmap command, creating meaningful training signals.

### Reverse Process: Iterative Denoising

The model learns `p(x_{t-1} | x_t, query)` - predicting the less noisy command given:

- Current noisy state `x_t`
- Natural language query (condition)

**Training Objective**:

```
L = -log p_θ(x_{t-1} | x_t, query)
  = CrossEntropy(model(x_t, query), x_{t-1})
```

Each training pair `(x_t, x_{t-1})` teaches the model to:

1. **Add missing flags** based on query intent
2. **Maintain correct syntax** and flag ordering
3. **Respect semantic constraints** (e.g., don't mix conflicting scan types)
4. **Preserve existing flags** from previous step

### Inference Algorithm

```python
def sample(query, max_steps=15):
    # Extract intent: which flags are allowed?
    allowed_flags = extract_intent(query)  # e.g., {'-sS', '-p 80,443', '-sV'}
    target = extract_target(query)          # e.g., '192.168.1.0/24'

    x = "nmap"  # Start from maximum noise

    for t in range(max_steps):
        # Model predicts next refinement
        x_next = model.generate("refine: " + x + " | query: " + query)

        # Enforce semantic constraints
        x_next = keep_only_allowed_flags(x_next, allowed_flags)
        x_next = add_target_if_missing(x_next, target)

        # Check convergence
        if converged(x, x_next):
            break

        x = x_next

    return x
```

**Convergence Criteria**:

- Exact string match: `x_t == x_{t-1}`
- Minimal change: ≤1 token difference
- Degradation detection: fewer flags than previous step

### Intent-Based Constraint Enforcement

To prevent hallucination, we extract constraints from the natural language query:

**Query**: "Stealth SYN scan on ports 80,443 with version detection on 192.168.1.0/24"

**Extracted Constraints**:

```python
{
    'scan_type': ['-sS'],              # "stealth" → SYN scan
    'ports': ['-p 80,443'],            # explicit ports
    'os_version': ['-sV'],             # "version detection"
    'target': '192.168.1.0/24'
}
```

**Enforcement**: After each model prediction, remove any flags not in allowed set and ensure target is present.

### Training Data Generation

From 4,638 clean commands, we generate **~32,000 training pairs**:

```python
for clean_command in dataset:
    sequence = generate_noise_sequence(clean_command)  # 7 steps
    # sequence = ["nmap", "nmap -sS target", ..., clean_command]

    for i in range(len(sequence) - 1):
        training_pairs.append({
            'input': f"refine: {sequence[i]} | query: {nl_query}",
            'target': sequence[i+1]
        })
```

**Data Augmentation**: Paraphrase queries ("Scan" → "Check", "Run" → "Execute") for +33% more data.

---

## Example Generation Trace

**Query**: "Scan all ports with OS detection on 192.168.1.0/24"

```
Step 0: nmap
  ↓ [Model adds scan type based on default]
Step 1: nmap 192.168.1.0/24
  ↓ [Model adds port specification from "all ports"]
Step 2: nmap -p- 192.168.1.0/24
  ↓ [Model adds OS detection from query]
Step 3: nmap -p- -O 192.168.1.0/24
  ↓ [Converged: no change]

Final: nmap -p- -O 192.168.1.0/24
```

**Why Diffusion Works Here**:

1. **Structured Output**: Commands have clear syntax rules - easier to learn incrementally
2. **Semantic Hierarchy**: Flags have natural ordering (scan → ports → detection)
3. **Error Correction**: Each step can fix mistakes from previous step
4. **Generalization**: Model learns flag relationships, not just memorization

---

## Running the System

```bash
# Train: python discrete_diffusion_nmap.py --mode train --epochs 20
# Inference: python discrete_diffusion_nmap.py --mode inference
# API: python agent_api_server.py
```

## Files

- **discrete_diffusion_nmap.py**: Main training/inference script
- **agent_api_server.py**: FastAPI REST endpoint
- **nmap_commands.json**: 4638 training examples
- **nmap_diffusion_checkpoint/**: Trained model (60M params)

---

Frontend – Nmap-AI Interface
📌 Overview

Interface utilisateur React / TypeScript permettant d’interagir avec le système multi-agents Nmap-AI.
Elle offre une expérience intuitive pour :

Soumettre des requêtes en langage naturel

Visualiser le pipeline multi-agents

Consulter les résultats détaillés des scans Nmap

🛠️ Technologies Stack

Framework : React 18+ (Vite)

Langage : TypeScript

Styling : CSS Modules / Tailwind CSS

State Management : React Hooks (useState, useEffect, etc.)

API Client : Fetch / Axios

📁 Structure du Projet

<img width="630" height="778" alt="image" src="https://github.com/user-attachments/assets/1eaa2353-0707-4e2a-86af-3b5e30bf6c6f" />

Fonctionnalités Principales
1️⃣ Dashboard (Dashboard.tsx)

       📊 Vue globale du système
       
       🤖 État des agents (Router, RAG, Diffusion, MCP)
       
       📈 Statistiques en temps réel
       
       🕒 Historique des commandes générées
       
       ⏱️ Graphiques de performance (temps de réponse par agent)

2️⃣ Router Page (RouterPage.tsx)

       🧠 Saisie de requêtes en langage naturel
       
       🔁 Visualisation du pipeline multi-agents :

       Comprehension → Complexity → Routing → Generation → Validation
       Résultats détaillés :

              Commande Nmap générée
              
              Niveau de complexité détecté
              
              Agent sélectionné (RAG / LoRA-T5 / Diffusion)
              
              Logs de validation MCP
              
              Résultats d’exécution

3️⃣ Composants Réutilisables

Layout.tsx : Structure globale (header, navigation)
       
RobotPipeline.tsx : Visualisation interactive du pipeline
       
⚙️ Installation & Démarrage
✅ Prérequis

       Node.js 18+
       npm      États en temps réel pour chaque agent
       
📦 Installation
cd frontend
npm install

▶️ Lancement en développement
npm run dev
Application accessible sur : 👉 http://localhost:3000
