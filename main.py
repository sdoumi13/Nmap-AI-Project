import os
import sys
import json
import yaml
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import httpx

# Ajout du chemin racine pour s'assurer que les imports fonctionnent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Chemin vers le fichier d'historique
HISTORY_FILE = "scan_history.json"

# --- CONFIGURATION DES SERVICES EXTERNES ---
# Architecture multi-services: chaque service tourne dans son propre terminal
COMPLEXITY_API_URL = "http://localhost:7000"  # Terminal 2: Complexity Classifier
MCP_AGENT5_URL = "http://localhost:5000"       # Terminal 1: Agent 5 MCP Server

# --- IMPORTS DES AGENTS LOCAUX ---
# On garde seulement les agents qui sont utilisés localement (pas de service séparé)
try:
    from agent_1_router.comprehension import ComprehensionAgent
    # Import des agents de génération (RAG et Diffusion) - utilisés localement
    from RAG.agent.rag_agent import NmapRagAgent
    from diffusion_models.discrete_diffusion_nmap import NmapDiscreteDiffusionLM, DiscreteDiffusionSampler
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    print("Vérifiez la structure de vos dossiers et les fichiers __init__.py")

app = FastAPI(title="Nmap-AI Backend API")

# --- CONFIGURATION CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# --- INITIALISATION GLOBALE DES AGENTS LOCAUX ---
print("🚀 Chargement des modèles IA et des Pipelines...")
print(f"📡 Services externes attendus:")
print(f"   - Complexity Classifier: {COMPLEXITY_API_URL}")
print(f"   - Agent 5 MCP Server: {MCP_AGENT5_URL}")

comp_agent = None
rag_agent = None
diffusion_model = None
diffusion_sampler = None

try:
    # Agent de compréhension (local - pas de service séparé)
    print("📖 Initialisation de l'agent de compréhension...")
    comp_agent = ComprehensionAgent()
    print("✅ Agent de compréhension prêt")
    
    # Agents de génération (locaux - utilisés directement par le router)
    try:
        print("📚 Initialisation de l'agent RAG...")
        rag_agent = NmapRagAgent()
        print("✅ Agent RAG prêt")
    except Exception as e:
        print(f"⚠️  Agent RAG non disponible: {e}")
    
    try:
        print("🎨 Initialisation du modèle Diffusion...")
        diffusion_model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
        diffusion_sampler = DiscreteDiffusionSampler(diffusion_model, max_steps=15)
        print("✅ Modèle Diffusion prêt")
    except Exception as e:
        print(f"⚠️  Modèle Diffusion non disponible: {e}")
    
    print("✅ Agents locaux initialisés !")
    print("⚠️  Assurez-vous que les services externes sont démarrés:")
    print("   Terminal 1: python agent_5_validation/run_agent5.py")
    print("   Terminal 2: python -m uvicorn agent_1_router.complexity:app --port 7000")
    print("   Terminal 3: ollama run llama3")
except Exception as e:
    print(f"❌ Erreur initialisation globale : {e}")

# Client HTTP pour les appels aux services externes
http_client = httpx.AsyncClient(timeout=60.0)

# --- MODÈLES DE DONNÉES ---
class QueryRequest(BaseModel):
    query: str
    target: str = "192.168.188.128"  # IP cible, peut être spécifiée par le frontend

# Nouveau modèle pour l'exécution de l'Agent 5
class ExecutionRequest(BaseModel):
    entry_id: str
    intent: str
    command: str
    target: str
    agent_name: str

# Modèle pour la génération de commande
class GenerateCommandRequest(BaseModel):
    query: str
    target: str = "192.168.188.128"
    agent_type: str = None  # "RAG" ou "DIFFUSION", si None sera déterminé automatiquement

# --- FONCTIONS UTILITAIRES ---

async def call_complexity_api(query: str) -> dict:
    """
    Appelle l'API Complexity Classifier (service externe sur port 7000).
    """
    try:
        response = await http_client.post(
            f"{COMPLEXITY_API_URL}/classify",
            json={"query": query, "user_id": "main-api"}
        )
        response.raise_for_status()
        data = response.json()
        
        # Convertir le format de réponse si nécessaire
        level_map = {"EASY": "Easy", "MEDIUM": "Medium", "HARD": "Hard"}
        return {
            "level": level_map.get(data.get("complexity", "MEDIUM"), "Medium"),
            "target_agent": "RAG" if data.get("complexity") == "EASY" else "DIFFUSION",
            "confidence": data.get("confidence", 0.5),
            "reason": data.get("reasoning", "Classification effectuée")
        }
    except httpx.HTTPError as e:
        print(f"⚠️  Erreur appel Complexity API: {e}")
        # Fallback: classification par défaut
        return {
            "level": "Medium",
            "target_agent": "DIFFUSION",
            "confidence": 0.5,
            "reason": f"Erreur API, fallback: {str(e)}"
        }

async def call_agent5_mcp(command: str, intent: str, target: str, agent_name: str) -> dict:
    """
    Appelle l'API Agent 5 MCP Server (service externe sur port 5000).
    """
    try:
        response = await http_client.post(
            f"{MCP_AGENT5_URL}/mcp/execute",
            json={
                "command": command,
                "intent": intent,
                "target": target,
                "agent_name": agent_name,
                "skip_sandbox": False
            }
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"⚠️  Erreur appel Agent 5 MCP: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service Agent 5 MCP non disponible: {str(e)}. Assurez-vous qu'il est démarré sur {MCP_AGENT5_URL}"
        )

async def generate_command_with_agent(query: str, target: str, agent_type: str) -> str:
    """
    Génère une commande nmap avec l'agent approprié (RAG ou Diffusion).
    
    Args:
        query: La requête utilisateur
        target: L'IP cible (optionnel)
        agent_type: "RAG" ou "DIFFUSION"
    
    Returns:
        La commande générée ou None en cas d'erreur
    """
    try:
        if agent_type == "RAG" and rag_agent is not None:
            result = rag_agent.process({
                "user_query": query,
                "extracted_ip": target
            })
            if result.get('status') == 'success':
                command = result.get('nmap_candidate', '')
                # Remplacer <TARGET> si présent
                if target and "<TARGET>" in command:
                    command = command.replace("<TARGET>", target)
                return command
        
        elif agent_type == "DIFFUSION" and diffusion_sampler is not None:
            result = diffusion_sampler.sample(query, verbose=False)
            command = result.get('final_command', '')
            # Remplacer <target> si présent
            if target and "<target>" in command:
                command = command.replace("<target>", target)
            return command
        
        # Fallback: utiliser la commande du comprehension agent si disponible
        return None
    except Exception as e:
        print(f"⚠️  Erreur génération commande ({agent_type}): {e}")
        return None

def save_to_history(query, analysis_result):
    """Initialise une entrée dans l'historique après l'analyse initiale."""
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    is_relevant = analysis_result.get("relevant", False)
    
    new_entry = {
        "id": str(len(history) + 1),
        "query": query,
        "best_match_command": analysis_result.get("best_match_command"),
        "generated_command": analysis_result.get("generated_command"),
        "generation_method": analysis_result.get("generation_method"),
        "complexity": analysis_result["analysis"]["level"] if is_relevant else "N/A",
        "status": "pending" if is_relevant else "failed",
        "timestamp": datetime.now().isoformat(),
        "target_agent": analysis_result["analysis"]["target_agent"] if is_relevant else None,
        "execution_report": None # Sera rempli par l'Agent 5
    }
    
    history.append(new_entry)
    with open(HISTORY_FILE, "w", encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    return new_entry["id"]

def update_history_with_report(entry_id, report):
    """Met à jour l'entrée avec le rapport détaillé de l'Agent 5."""
    if not os.path.exists(HISTORY_FILE):
        return
    
    with open(HISTORY_FILE, "r", encoding='utf-8') as f:
        history = json.load(f)
    
    for entry in history:
        if entry["id"] == entry_id:
            entry["status"] = "completed" if report["final_status"] == "success" else "failed"
            entry["execution_report"] = report
            # Met à jour la commande si elle a été auto-corrigée
            if report.get("stages", {}).get("self_correction", {}).get("applied"):
                entry["best_match_command"] = report["stages"]["self_correction"]["final_command"]
            break
            
    with open(HISTORY_FILE, "w", encoding='utf-8') as f:
        json.dump(history, f, indent=4)

# --- ENDPOINTS API ---

@app.post("/api/analyze")
async def analyze_and_route(request: QueryRequest):
    """
    Pipeline complet STEP 1-3:
    STEP 1: Comprehension Check
    STEP 2: Complexity Classification  
    STEP 3: Command Generation (RAG or Diffusion)
    """
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="La requête est vide.")

    try:
        # STEP 1: Comprehension Check
        comp_result = comp_agent.analyze(user_query)
        
        if not comp_result['relevant']:
            result = {
                "relevant": False, 
                "reason": comp_result['reason'], 
                "status": "failed", 
                "analysis": {"level": "N/A", "target_agent": None, "confidence": 0.0, "reason": comp_result['reason']}
            }
            save_to_history(user_query, result)
            return result

        # STEP 2: Complexity Classification (appel HTTP vers service externe)
        routing = await call_complexity_api(user_query)
        level = routing['level']
        
        # Déterminer l'agent selon la complexité (comme dans run_router.py)
        # Easy -> RAG, Medium/Hard -> Diffusion
        agent_choice = "RAG" if level == "Easy" else "DIFFUSION"
        
        # Utiliser le target fourni par le frontend ou la valeur par défaut
        target = request.target if request.target else "192.168.188.128"
        
        # STEP 3: Command Generation
        generated_command = await generate_command_with_agent(user_query, target, agent_choice)
        
        # Utiliser la commande générée ou fallback sur best_match
        final_command = generated_command or comp_result['best_match'].get('command') if comp_result.get('best_match') else None
        
        result = {
            "relevant": True,
            "best_match_command": final_command,
            "generated_command": generated_command,
            "generation_method": agent_choice if generated_command else "fallback",
            "analysis": {
                "level": level,
                "target_agent": agent_choice,
                "confidence": routing.get('confidence', 0.5),
                "reason": routing['reason']
            }
        }
        
        # Sauvegarde et récupération de l'ID pour le frontend
        entry_id = save_to_history(user_query, result)
        result["entry_id"] = entry_id
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def call_agent5_validate(command: str, intent: str, agent_name: str) -> dict:
    """
    Appelle l'API Agent 5 MCP Server pour la validation seule (service externe sur port 5000).
    """
    try:
        response = await http_client.post(
            f"{MCP_AGENT5_URL}/mcp/validate",
            json={
                "command": command,
                "intent": intent,
                "agent_name": agent_name
            }
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"⚠️  Erreur appel Agent 5 MCP Validation: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service Agent 5 MCP non disponible: {str(e)}. Assurez-vous qu'il est démarré sur {MCP_AGENT5_URL}"
        )

@app.post("/api/validate")
async def validate_command(request: ExecutionRequest):
    """
    Agent 5 : Validation seule d'une commande.
    Appelle le service Agent 5 MCP Server (port 5000) pour la validation.
    """
    try:
        # Appel HTTP vers le service Agent 5 MCP (service externe)
        validation_result = await call_agent5_validate(
            command=request.command,
            intent=request.intent,
            agent_name=request.agent_name
        )
        
        return validation_result
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 Erreur Validation Agent 5: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de validation: {str(e)}")

@app.post("/api/execute")
async def execute_pipeline(request: ExecutionRequest):
    """
    Agent 5 : Pipeline complet de Validation et Exécution.
    Appelle le service Agent 5 MCP Server (port 5000).
    """
    try:
        # Appel HTTP vers le service Agent 5 MCP (service externe)
        report = await call_agent5_mcp(
            command=request.command,
            intent=request.intent,
            target=request.target,
            agent_name=request.agent_name
        )
        
        # Mise à jour de la persistence
        update_history_with_report(request.entry_id, report)
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 Erreur Pipeline Agent 5: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'exécution: {str(e)}")

@app.get("/api/history")
async def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    return []

@app.post("/api/generate")
async def generate_command_endpoint(request: GenerateCommandRequest):
    """
    Endpoint dédié pour la génération de commande (STEP 3).
    Peut être appelé séparément après l'analyse.
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="La requête est vide.")
        
        # Si agent_type n'est pas spécifié, déterminer via complexity (appel HTTP)
        agent_type = request.agent_type
        if not agent_type:
            routing = await call_complexity_api(query)
            level = routing['level']
            agent_type = "RAG" if level == "Easy" else "DIFFUSION"
        
        generated_command = await generate_command_with_agent(query, request.target, agent_type)
        
        if not generated_command:
            raise HTTPException(status_code=500, detail=f"Impossible de générer la commande avec l'agent {agent_type}")
        
        return {
            "command": generated_command,
            "agent_type": agent_type,
            "query": query,
            "target": request.target
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Vérifie l'état de tous les services (local et externes)."""
    health_status = {
        "status": "online",
        "local_agents": {
            "comprehension_ready": comp_agent is not None,
            "rag_ready": rag_agent is not None,
            "diffusion_ready": diffusion_sampler is not None
        },
        "external_services": {}
    }
    
    # Vérifier Complexity API
    try:
        response = await http_client.get(f"{COMPLEXITY_API_URL}/health", timeout=5.0)
        health_status["external_services"]["complexity_api"] = {
            "status": "online" if response.status_code == 200 else "error",
            "url": COMPLEXITY_API_URL
        }
    except Exception as e:
        health_status["external_services"]["complexity_api"] = {
            "status": "offline",
            "url": COMPLEXITY_API_URL,
            "error": str(e)
        }
    
    # Vérifier Agent 5 MCP
    try:
        response = await http_client.get(f"{MCP_AGENT5_URL}/health", timeout=5.0)
        health_status["external_services"]["agent5_mcp"] = {
            "status": "online" if response.status_code == 200 else "error",
            "url": MCP_AGENT5_URL
        }
    except Exception as e:
        health_status["external_services"]["agent5_mcp"] = {
            "status": "offline",
            "url": MCP_AGENT5_URL,
            "error": str(e)
        }
    
    return health_status

@app.on_event("shutdown")
async def shutdown_event():
    """Fermeture propre du client HTTP."""
    await http_client.aclose()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 NMAP-AI BACKEND API (Port 8000)")
    print("="*70)
    print("\n⚠️  ARCHITECTURE MULTI-SERVICES:")
    print("   Ce serveur nécessite que les services suivants soient démarrés:")
    print(f"   1. Agent 5 MCP Server: {MCP_AGENT5_URL}")
    print(f"   2. Complexity Classifier: {COMPLEXITY_API_URL}")
    print("   3. Ollama LLM (pour RAG): ollama run llama3")
    print("\n" + "="*70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)