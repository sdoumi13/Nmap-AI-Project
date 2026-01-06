# Flux Complet: RAG Distant → Agent 5 MCP

## Vue d'ensemble

```
┌─ Salma (192.168.1.169) ─┐     ┌─ Abdellah (192.168.1.218) ─┐
│                                 │     │                                      │
│  User Query                      │     │                                      │
│      ↓                           │     │                                      │
│  [Agent 1 Router]                │     │                                      │
│      ↓                           │     │                                      │
│  Complexity: EASY                │     │                                      │
│      ↓                           │     │                                      │
│  DistributedRAGClient            │     │                                      │
│      │                           │     │                                      │
│      │  HTTP POST                │     │                                      │
│      └─────────────────────────────────→ [Agent 2 Server :8000]              │
│                                 │     │       ↓                              │
│                                 │     │   NmapRagAgent.process()             │
│                                 │     │       ↓                              │
│                                 │     │   MCPAgent5Client.validate()         │
│                                 │     │       │                              │
│                                 │     │       │  HTTP POST (MCP)             │
│      [Agent 5 MCP Server :5000]←──────────────│ /mcp/validate                │
│      ↓                          │     │       ↑                              │
│  HybridValidator                │     │       │                              │
│      ↓                          │     │  validation_result                   │
│  validation_response            │     │       ↑                              │
│      ↓                          │     │  HTTP 200 OK                         │
│      │  HTTP 200 OK             │     │       │                              │
│      └──────────────────────────→────→ Returns to Client                     │
│                                 │     │       ↓                              │
│                                 │     │  Build response JSON                 │
│                                 │     │       ↓                              │
│                                 │     │  HTTP 200 OK to Router               │
│      ↓                          │     │  (validation + command)              │
│  Receives validation+command    │     │                                      │
│      ↓                          │     │                                      │
│  DistributedRAGClient resolves  │     │                                      │
│      ↓                          │     │                                      │
│  Ready for MCP Execute          │     │                                      │
│                                 │     │                                      │
└─────────────────────────────────┘     └──────────────────────────────────────┘
```

---

## Étapes détaillées

### ÉTAPE 1: Router détecte complexité EASY
**Fichier**: `agent_1_router/run_router.py`

```python
# USER QUERY: "scan tcp port 22"
complexity_result = self.complexity_agent.classify("scan tcp port 22")
# → {'level': 'Easy', 'confidence': 0.95}

# Décision: utiliser RAG (EASY)
```

### ÉTAPE 2: Router envoie requête au RAG distant

**Fichier**: `agent_1_router/run_router.py` - méthode `_generate_rag_command()`

```python
async def _generate_rag_command(self, query: str, target: str) -> str:
    client = DistributedRAGClient(rag_url="http://192.168.1.218:8000")
    result = await client.generate_command(query=query, target=target)
    # Appel HTTP vers le serveur du collègue
    return result.get('command')
```

**HTTP REQUEST**:
```
POST http://192.168.1.218:8000/generate_command HTTP/1.1
Content-Type: application/json

{
    "query": "scan tcp port 22",
    "target": "192.168.188.128",
    "source_agent": "router-agent-1"
}
```

---

### ÉTAPE 3: Agent 2 du collègue reçoit la requête

**Fichier**: `RAG/agent2_server.py` (sur machine 192.168.1.218)

```python
@app.post("/generate_command")
async def generate_command(request: GenerateCommandRequest):
    # 1. Crée une instance de NmapRagAgent
    rag_agent = NmapRagAgent(dataset_path="nmap_dataset.json")
    
    # 2. Traite la requête
    rag_response = rag_agent.process(
        query=request.query,
        target=request.target
    )
    # → rag_response = {"command": "nmap -sT -p 22 192.168.188.128", ...}
    
    # 3. Valide via MCP (appelle votre Agent 5)
    # → Pour valider la commande générée
```

**Logique RAG**:
- Récupère les embeddings de `nmap_dataset.json`
- Cherche les commandes similaires dans la Chroma DB
- Génère une nouvelle commande via Ollama

**Résultat RAG**:
```json
{
    "command": "nmap -sT -p 22 192.168.188.128",
    "reasoning": "TCP port scan on port 22 (SSH)",
    "confidence": 0.92,
    "matched_examples": ["nmap -sT 192.168.1.1", "nmap -p 22"]
}
```

---

### ÉTAPE 4: Agent 2 valide via MCP (appelle votre Agent 5)

**Fichier**: `RAG/mcp_agent5_client.py` (sur machine 192.168.1.218)

```python
class MCPAgent5ClientSync:
    def validate_command(self, command: str, intent: str, target: str, source_agent: str):
        """Appelle l'endpoint MCP de votre Agent 5"""
        
        payload = {
            "type": "mcp",
            "method": "validate",
            "params": {
                "command": command,
                "intent": intent,
                "target": target,
                "source_agent": source_agent,
                "validation_type": "hybrid"
            }
        }
        
        # HTTP POST vers votre machine
        response = requests.post(
            "http://192.168.1.169:5000/mcp/validate",
            json=payload,
            timeout=30
        )
        return response.json()
```

**HTTP REQUEST** (du collègue vers VOUS):
```
POST http://192.168.1.169:5000/mcp/validate HTTP/1.1
Content-Type: application/json

{
    "type": "mcp",
    "method": "validate",
    "params": {
        "command": "nmap -sT -p 22 192.168.188.128",
        "intent": "scan tcp port 22",
        "target": "192.168.188.128",
        "source_agent": "rag-agent-2",
        "validation_type": "hybrid"
    }
}
```

---

### ÉTAPE 5: Votre Agent 5 valide la commande

**Fichier**: `agent_5_validation/mcp_tools/mcp_server.py`

```python
@app.post("/mcp/validate")
async def mcp_validate(request: MCPValidateRequest):
    """
    Endpoint MCP pour valider une commande
    """
    
    # 1. Validation sémantique
    semantic_result = validator.validate_semantic(
        command=request.command,
        intent=request.intent,
        target=request.target
    )
    # → {'valid': True, 'score': 92, 'method': 'semantic'}
    
    # 2. Validation LLM (si score sémantique < 80)
    if semantic_result['score'] < 80:
        llm_result = validator.validate_llm(
            command=request.command,
            intent=request.intent
        )
        # → {'valid': True, 'score': 88, 'method': 'llm'}
        final_score = max(semantic_result['score'], llm_result['score'])
    else:
        final_score = semantic_result['score']
    
    # 3. Retourne la réponse MCP
    return {
        "type": "mcp",
        "status": "success",
        "result": {
            "valid": final_score >= 70,
            "score": final_score,
            "method_used": "hybrid",
            "semantic_score": semantic_result['score'],
            "source_agent": request.source_agent
        }
    }
```

**HTTP RESPONSE**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
    "type": "mcp",
    "status": "success",
    "result": {
        "valid": true,
        "score": 92,
        "method_used": "hybrid",
        "semantic_score": 92,
        "source_agent": "rag-agent-2"
    }
}
```

---

### ÉTAPE 6: Agent 2 reçoit la validation et retourne au Router

**Fichier**: `RAG/agent2_server.py`

```python
@app.post("/generate_command")
async def generate_command(request: GenerateCommandRequest):
    # ... étapes 1-2 ...
    
    # 3. Valide via MCP
    mcp_client = MCPAgent5ClientSync(agent5_url="http://192.168.1.169:5000")
    validation_result = mcp_client.validate_command(
        command=rag_response['command'],
        intent=request.query,
        target=request.target,
        source_agent="rag-agent-2"
    )
    
    # 4. Retourne la réponse au Router
    return {
        "status": "success",
        "command": rag_response['command'],
        "validation": validation_result['result'],  # Includescore MCP
        "source": "rag-agent-2",
        "timestamp": datetime.now().isoformat()
    }
```

**HTTP RESPONSE** au Router:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
    "status": "success",
    "command": "nmap -sT -p 22 192.168.188.128",
    "validation": {
        "valid": true,
        "score": 92,
        "method_used": "hybrid",
        "semantic_score": 92,
        "source_agent": "rag-agent-2"
    },
    "source": "rag-agent-2",
    "timestamp": "2025-01-15T12:34:56.789Z"
}
```

---

### ÉTAPE 7: Router reçoit la commande validée

**Fichier**: `agent_1_router/run_router.py`

```python
async def _generate_rag_command(self, query: str, target: str) -> str:
    client = DistributedRAGClient(rag_url="http://192.168.1.218:8000")
    result = await client.generate_command(query=query, target=target)
    
    # result =
    # {
    #     "status": "success",
    #     "command": "nmap -sT -p 22 192.168.188.128",
    #     "validation": {"valid": true, "score": 92, ...},
    #     ...
    # }
    
    if result.get('status') == 'success':
        command = result.get('command')
        validation = result.get('validation', {})
        
        print(f"✅ Command: {command}")
        print(f"   Validated with MCP: Score {validation.get('score')}/100")
        
        return command
    
    return None
```

---

### ÉTAPE 8: Router envoie pour exécution

```python
# Dans router.route()
mcp_result = await self.mcp_client.execute_command(
    command=command,
    intent=user_query,
    target=target,
    agent_name="rag"  # Indique que la commande vient du RAG
)
```

**HTTP REQUEST** (MCP Execute):
```
POST http://localhost:5000/mcp/execute HTTP/1.1
Content-Type: application/json

{
    "command": "nmap -sT -p 22 192.168.188.128",
    "intent": "scan tcp port 22",
    "target": "192.168.188.128",
    "agent_name": "rag",
    "skip_sandbox": false
}
```

---

## Résumé du flux MCP

```
ROUTER (IP: 192.168.1.169:7000)
    │
    ├─ Detect EASY complexity
    │
    ├─ HTTP POST → Colleague RAG (http://192.168.1.218:8000/generate_command)
    │
    └─ Colleague RAG (IP: 192.168.1.218:8000)
        │
        ├─ NmapRagAgent.process()
        │
        ├─ HTTP POST → Your Agent 5 MCP (http://192.168.1.169:5000/mcp/validate)
        │
        └─ Your Agent 5 (IP: 192.168.1.169:5000)
            │
            ├─ HybridValidator validates command
            │
            └─ HTTP 200 OK → Back to Colleague RAG
                │
                └─ HTTP 200 OK → Back to Router
                    │
                    └─ Router receives command + validation score
                        │
                        └─ HTTP POST → Agent 5 MCP (http://localhost:5000/mcp/execute)
                            │
                            └─ Full execution pipeline
```

---

## Dépannage

### Problème: Router ne peut pas atteindre RAG distant
```
❌ Connection refused on 192.168.1.218:8000
```

**Solutions**:
1. Vérifier que le serveur Agent 2 est démarré
   ```bash
   # Sur la machine du collègue
   python RAG/agent2_server.py
   ```

2. Vérifier la connectivité réseau
   ```bash
   ping 192.168.1.218
   ```

3. Vérifier le firewall (port 8000 doit être ouvert)

### Problème: MCP validation échoue
```
{"valid": false, "score": 45, "method_used": "hybrid"}
```

**Solutions**:
1. Vérifier que votre Agent 5 est accessible
   ```bash
   curl http://localhost:5000/health
   ```

2. Vérifier les scores:
   - Score < 70 = INVALIDE
   - Score 70-85 = DOUTEUX (auto-correction nécessaire)
   - Score > 85 = VALIDE

3. Vérifier les logs du validateur

---

## Fichiers impliqués

### Sur VOTRE machine (192.168.1.169)
- `agent_1_router/run_router.py` - Router principal
- `agent_5_validation/mcp_tools/mcp_server.py` - Serveur MCP
- `agent_5_validation/validation/hybrid_validator.py` - Logique de validation

### Sur la MACHINE DU COLLÈGUE (192.168.1.218)
- `RAG/agent2_server.py` - Serveur HTTP qui expose le RAG
- `RAG/mcp_agent5_client.py` - Client MCP pour appeler votre Agent 5
- `RAG/agent/rag_agent.py` - Logique du RAG (NmapRagAgent)
- `RAG/nmap_dataset.json` - Données d'entraînement du RAG
- `RAG/chroma_db_local/` - Cache vectoriel

---

## Configuration requise

### Machine du collègue (192.168.1.218:8000)
```bash
# Démarrer le serveur Agent 2
python RAG/agent2_server.py

# Dépendances requises:
# - httpx (client HTTP async)
# - requests (client HTTP sync)
# - ollama (pour génération de commandes)
# - chromadb (pour recherche vectorielle)
```

### Votre machine (192.168.1.169)
```bash
# Démarrer le serveur Agent 5 MCP
python agent_5_validation/run_agent5.py

# Démarrer le Router
python run_router_main.py

# Dépendances requises:
# - fastapi
# - httpx
# - torch (pour validation)
# - docker (pour sandbox)
```

---

## Flux complet (résumé)

```
1. Utilisateur: "scan tcp port 22"
2. Router: Complexité → EASY
3. Router → RAG distant (HTTP POST)
4. RAG: Génère "nmap -sT -p 22 192.168.188.128"
5. RAG → Agent 5 MCP (HTTP POST validate)
6. Agent 5: Valide (Score: 92/100)
7. Agent 5 → RAG (HTTP 200)
8. RAG → Router (HTTP 200 + validation)
9. Router → Agent 5 MCP (HTTP POST execute)
10. Agent 5: Exécute (Sandbox + VM)
11. Résultat final retourné à l'utilisateur
```

C'est le flux MCP complet! 🚀
