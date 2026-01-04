"""
complexity_api.py
API REST utilisant ComplexityAgent local
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Literal
import sys
import os
from pathlib import Path

# Setup imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from agent_1_router.complexity import ComplexityAgent

app = FastAPI(
    title="Nmap Complexity Classifier API",
    description="Routes queries to appropriate agent (RAG or Diffusion)",
    version="1.0"
)

# ============= MODELS =============

class ComplexityRequest(BaseModel):
    query: str
    user_id: str = "anonymous"

class ComplexityResponse(BaseModel):
    query: str
    complexity: Literal["EASY", "MEDIUM", "HARD"]
    confidence: float
    recommended_agent: Literal["RAG", "DIFFUSION"]
    reasoning: str

# ============= GLOBAL CLASSIFIER =============

complexity_agent = None

@app.on_event("startup")
def startup():
    global complexity_agent
    print("🔍 Loading ComplexityAgent...")
    complexity_agent = ComplexityAgent(
        finetuning_filename='finetuning_corpus_detailed.json',
        diffusion_filename='diffusion_corpus_detailed.json'
    )
    print("✅ ComplexityAgent ready")

# ============= ENDPOINTS =============

@app.get("/")
def root():
    return {
        "service": "Nmap Complexity Classifier API",
        "version": "1.0",
        "endpoints": {
            "/classify": "POST - Classify query complexity",
            "/health": "GET - Service health check"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy" if complexity_agent else "initializing"}

@app.post("/classify", response_model=ComplexityResponse)
def classify_query(request: ComplexityRequest):
    if not complexity_agent:
        raise HTTPException(status_code=503, detail="Classifier not ready")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = complexity_agent.classify(request.query)
        
        # Map level to API format
        level_map = {"Easy": "EASY", "Medium": "MEDIUM", "Hard": "HARD"}
        agent_map = {
            "Easy": "RAG",
            "Medium": "DIFFUSION", 
            "Hard": "DIFFUSION"
        }
        
        complexity = level_map.get(result['level'], "MEDIUM")
        
        return ComplexityResponse(
            query=request.query,
            complexity=complexity,
            confidence=result['confidence'],
            recommended_agent=agent_map.get(result['level'], "DIFFUSION"),
            reasoning=result['reason']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════════╗
    ║   NMAP COMPLEXITY CLASSIFIER API          ║
    ║   Easy → RAG | Medium/Hard → Diffusion    ║
    ╚════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="info")