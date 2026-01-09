# Nmap-AI Project

## Team Members

- **Role 1:** DOUMI SALMA — Complexity Agent & Validation (MCP) Agent
- **Role 2:** — AFROUKH ABDELLAH \_ Nmap Discrete Diffusion Agent
- **Role 3:** — NACIRI AYMANE - FRONTEND DEVELOPPER + BACKEND/FRONTEND APIS
- **Role 4:** — Aymane Moutmaine — RAG Agent
- **Role 5:** — BAY BAY BADR - Fine-tuning Agent

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

| Étape                | Fonction                | Objectif                                                           |
| -------------------- | ----------------------- | ------------------------------------------------------------------ |
| **1. Comprehension** | Filtre les requêtes     | Rejeter le bruit non-Nmap                                          |
| **2. Complexity**    | Classifie la difficulté | Easy / Medium / Hard                                               |
| **3. Routing**       | Sélectionne l'agent     | RAG (simple) /LoRA-fine-tuned T5-small-Phi-4 /Diffusion (complexe) |
| **4. Execution**     | Envoie à Agent 5        | Validation + Correction + Sandbox + VM                             |

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
# Agent 5: Validation & Self-Correction Architecture

## MCP Protocol 

### Overview of the MCP Protocol

The **MCP (Model Context Protocol)** is the communication backbone used to securely and consistently exchange data between **Agent 1 (Router)** and **Agent 5 (Validation & Execution Agent)**.

In the Nmap-AI system, MCP acts as a **control and governance layer** that ensures every generated Nmap command is:
- Semantically valid
- Security-compliant
- Safely executable
- Fully traceable

MCP is **not a simple API call**, but a structured protocol that defines:
- Message format
- Validation states
- Execution permissions
- Feedback loops

---

![Validation & Self-Correction Flow](/Annexe//validation+self-correction.png)

---

##  System Architecture

```
agent5_validation+self-correction-sandboxing-testVM/
├── __init__.py
├── validation/
│   ├── __init__.py
│   ├── semantic_validator.py      # Step 1: Rule-based validation
│   ├── llm_judge.py                # Step 2: LLM-based validation
│   └── hybrid_validator.py         # Step 3: Combined validation
├── mcp_tools/
│   ├── __init__.py
│   └── mcp_server.py              # Step 4-5-6: MCP server interface
├── execution/
│   ├── __init__.py
│   ├── sandbox_executor.py         # Step 7: Docker sandbox
│   └── vm_executor.py              # Step 8: VM SSH execution
├── self_correction/
│   ├── __init__.py
│   └── corrector.py                # Step 9: Self-correction agent
├── run_agent5.py                   # Step 10: Main orchestrator
└── agent5_config.yaml              # Configuration file
```

### Running the System

```bash
# Start Agent 5 MCP Server (Port 5002)
python run_agent5.py

# Or start MCP server independently
python mcp_tools/mcp_server.py
```

---

##  Validation Pipeline

### 1. Semantic Validator (Rule-Based)

**Purpose**: Fast, deterministic validation using regex patterns and semantic rules.

**Key Features**:
-  Detects dangerous commands (rm -rf, fork bombs)
-  Validates root privilege requirements
-  Checks target presence (IP/domain)
-  Identifies flag conflicts (-sS vs -sT)
-  Validates port syntax

---

### 2. LLM Judge (Qwen2.5-Coder-3B / Mistral-7B)
![LLM](/Annexe//LLM.png)

**Purpose**: Intelligent validation using language models for complex cases.

**Models Used**:
- **Qwen2.5-Coder-3B** (Port 1234) - Primary LLM judge
- **Mistral-7B** (Fallback) - Alternative validation

**Features**:
-  Context-aware validation
-  Intent matching verification
-  Confidence scoring
- 💡 Detailed reasoning

---

### 3. Hybrid Validator (Combined Approach)

**Purpose**: Best of both worlds - fast semantic rules + intelligent LLM validation.

**Decision Matrix**:
| Semantic Score | LLM Score | Final Status |
|----------------|-----------|--------------|
| 100            | -         | VALID      |
| 70 (root err)  | -         | RECOVERABLE 🔧 |
| < 50           | -         | INVALID ❌   |
| 60-79          | 80-100    | VALID      |
| 60-79          | 50-79     | RECOVERABLE 🔧 |

---
![Validation](/Annexe//Validation.png)

## 🔧 Self-Correction Mechanism

### Corrector Algorithm

**Purpose**: Automatically fix recoverable command errors.

**Flow Chart**:
```
┌─────────────────────┐
│  Invalid Command    │
│  (RECOVERABLE)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Analyze Error Type  │
└──────────┬──────────┘
           │
           ├─► Root Privilege Error    → Add 'sudo'
           ├─► Missing Target          → Add IP/TARGET
           ├─► Flag Conflict           → Remove conflicting flag
           ├─► Invalid Port Syntax     → Fix comma separation
           ├─► Missing Required Flag   → Add based on intent
           ├─► Invalid Decoy Format    → Fix to RND:10
           └─► Multiple Timing Flags   → Keep only one
           
           ▼
┌─────────────────────┐
│  Apply Fix          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Re-Validate        │
└──────────┬──────────┘
           │
           ├─► VALID (score ≥ 75)     →  Success
           ├─► Still Invalid          →  Retry (max 2)
           └─► Max Retries            →  Return best attempt
```

---

##  Security Features

1. **Dangerous Command Detection**: Blocks rm, dd, fork bombs
2. **Docker Isolation**: Sandboxed execution before VM
3. **SSH Hardening**: Secure VM communication
4. **Input Validation**: Multiple layers of checks
5. **Rate Limiting**: Prevents abuse (30 req/min)

---

> **MCP transforms LLM-generated commands into controlled, auditable, and secure operations.**  
It is the cornerstone that makes Nmap-AI suitable for **real-world cybersecurity environments**.

##  References

- Nmap Documentation: https://nmap.org/book/
- Docker Security: https://docs.docker.com/engine/security/
- LLM Validation Papers
- Semantic Validation 

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

 Installation
cd frontend
npm install

 Lancement en développement
npm run dev
Application accessible sur : 👉 http://localhost:3000

---

# Nmap RAG Agent

## Overview

Generates Nmap commands from natural language using Retrieval-Augmented Generation (RAG) with:

- HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Chroma vector database (persisted locally)
- Ollama LLM (`llama3:8b`) for command synthesis

## API Endpoints

- POST `/generate_command` → Generate Nmap command from `query` and `target`
- GET `/health` → Service health check

## Run Server

Start the FastAPI server listening on your LAN:

```bash
python RAG/server.py
```

or with Uvicorn:

```bash
uvicorn RAG.server:app --host 0.0.0.0 --port 8000
```

## Test from another machine

Use your machine IP (example: `192.168.1.141`):

```bash
curl -X POST http://192.168.1.141:8000/generate_command \
  -H "Content-Type: application/json" \
  -d '{"query":"scan all open ports","target":"192.168.1.1"}'
```

PowerShell alternative:

```powershell
Invoke-RestMethod -Uri "http://192.168.1.141:8000/generate_command" -Method POST -ContentType "application/json" -Body '{"query":"scan all open ports","target":"192.168.1.1"}'
```

Expected response:

```json
{
  "status": "success",
  "command": "nmap -p- 192.168.1.1",
  "intent": "scan all open ports",
  "target": "192.168.1.1",
  "agent": "RAG",
  "confidence": 0.8
}
```

# 🤖 Fine-tuned Agent - Phi3 mini Model ~3.8B parameters

An intelligent Fine-Tuned Agent designed to translate natural language requests into precise, syntax-accurate `nmap` commands. This agent serves as the "Tactical Specialist" in the multi-agent ecosystem.

##  Model Architecture

- **Base Model**: Phi-3 Mini (~3.8B parameters)
- **Fine-Tuning**: Trained using LoRA (Low-Rank Adaptation) for domain-specific mastery of network security syntax
- **Optimization**: 4-bit NF4 Quantization via `bitsandbytes`, enabling high-speed inference on consumer-grade GPUs (~4-6GB VRAM)

## 🚀 Key Features

- **Contextual Translation**: Converts complex scanning intents (e.g., "Find all web servers") into optimized flags (`nmap -p 80,443...`)
- **FastAPI Integration**: Provides a high-performance REST API for seamless communication with other agents
- **Hardware Accelerated**: Full CUDA support for near-instant command generation (~1-2s)

## 🛠 Quick Start

1. **Dependencies**:

```bash
   pip install torch transformers peft fastapi bitsandbytes
```

2. **Model Setup**:

   - Place Phi-3 base weights in `C:\models\phi3_mini`
   - Place LoRA adapters in `./phi3-nmap-results`

3. **Launch**:

```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📡 Agent API

- **Endpoint**: `POST /generate`
- **Input**:

```json
{ "prompt": "Scan 192.168.1.0/24 for OS version" }
```

- **Output**:

```json
{ "nmap_command": "nmap -O 192.168.1.0/24" }
```

##  Repository Structure

```
.
├── app.py                          # FastAPI agent interface
├── phi3-nmap-results/              # Pre-trained LoRA adapter checkpoints
├── nmap_dataset_augmented.json     # Curated cybersecurity training data
├── Finetuning Script.ipynb         # Training code
└── app.py                          # Core inference logic for the fine-tuned model
```

##  Usage Example

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "Find all web servers with SSL on network 10.0.0.0/24"}
)

print(response.json()["nmap_command"])
# Output: nmap -p 443 --script ssl-cert 10.0.0.0/24
```

##  Training Details

- **Dataset**: Custom cybersecurity corpus with 10,000+ nmap command pairs
- **Training Duration**: ~2 hours on RTX ADA 2000 8GB Vram
- **LoRA Parameters**: r=16, alpha=32, dropout=0.05
- **Validation Accuracy**: 94.2% syntax correctness

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  Disclaimer

This tool is intended for authorized security testing only. Always obtain proper authorization before scanning networks.

---

