# 🚀 GUIDE DE LANCEMENT DU PROJET NMAP-AI

## 📋 ORDRE DE DÉMARRAGE
```

Copiez-collez ces commandes dans 5 terminaux séparés :

### Terminal 1:
```bash
python -m uvicorn agent_5_validation.mcp_tools.mcp_server:app --port 5000 --reload
```

### Terminal 2:
```bash
python -m uvicorn agent_1_router.complexity:app --port 7000 --reload
```

### Terminal 3:
```bash
ollama run llama3
```

### Terminal 4:
```bash
python main.py
```

### Terminal 5:
```bash
cd frontend && npm run dev
```

---

## ✅ CHECKLIST DE VÉRIFICATION

Avant de commencer à utiliser le projet, vérifiez que :

- [ ] Terminal 1: Agent 5 MCP répond sur http://localhost:5000/health
- [ ] Terminal 2: Complexity API répond sur http://localhost:7000/health
- [ ] Terminal 3: Ollama est actif (modèle llama3 chargé)
- [ ] Terminal 4: Main API répond sur http://localhost:8000/api/health
- [ ] Terminal 5: Frontend accessible sur http://localhost:3000

---
