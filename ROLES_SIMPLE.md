# ⚡ RÉSUMÉ - RÔLES ET LANCEMENT

## 3 COMPOSANTS CLÉS

### 1. **MCP CLIENT** (dans Router)
- **Rôle:** Envoie la commande générée au serveur Agent 5
- **Lancer?** ❌ NON - C'est dans le Router
- **Type:** Classe Python (module interne)

### 2. **RAG AGENT** (pour requêtes FACILES)
- **Rôle:** Génère commandes en cherchant examples dans ChromaDB
- **Lancer?** ❌ NON - Appelé par Router
- **Type:** Module Python (internal)

### 3. **DIFFUSION MODEL** (pour requêtes COMPLEXES)
- **Rôle:** Génère commandes avec modèle T5 diffusion
- **Lancer?** ❌ NON - Appelé par Router
- **Type:** Module Python (interne)

---

## 🎯 CE QU'IL FAUT LANCER (4 SERVICES)

```
Terminal 1: python agent_5_validation\run_agent5.py
            ↳ Agent 5 MCP Server (validation + sandbox + VM)

Terminal 2: python -m uvicorn agent_1_router.complexity:app --port 7000
            ↳ Complexity Classifier (EASY/MEDIUM/HARD)

Terminal 3: ollama run llama3
            ↳ LLM Ollama (utilisé par RAG et Diffusion)

Terminal 4: python run_router_main.py
            ↳ Router Agent (interface user - c'est ici qu'on rentre les requêtes)
```

---

## 🔄 FLUX RÉSUMÉ

```
USER: "scan port 80"
  ↓ Router
  ├─ Complexity: EASY
  ├─ Appelle: RAG Agent
  ├─ RAG génère: "nmap -p 80 TARGET"
  ├─ MCP Client envoie à Agent 5
  ├─ Agent 5: Validate → Correct → Sandbox → VM
  └─ Résultats au user
```

---

## 📋 LANCEMENT RAPIDE

Ouvrir 4 terminaux et exécuter:

```bash
# Terminal 1
python agent_5_validation\run_agent5.py

# Terminal 2
python -m uvicorn agent_1_router.complexity:app --port 7000

# Terminal 3
ollama run llama3

# Terminal 4
python run_router_main.py
# Puis: ROUTER > votre requête
```

---

## ✅ APRÈS LA CORRECTION

**run_agent5.py:**
- ✅ Tests statiques supprimés
- ✅ Attend juste les commandes du Router
- ✅ Pas de main() automatique

**Flux:**
- ✅ Router → Classification → RAG/Diffusion → Commande → MCP Client → Agent 5 → Résultats

---

**Status:** ✅ COMPLET - Tout expliqué et modifié
