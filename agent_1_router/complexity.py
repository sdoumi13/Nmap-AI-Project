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
import json

# Setup imports
current_dir = Path(__file__).resolve().parent
project_root = Path(current_dir).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============= COMPLEXITY AGENT CLASS =============

class ComplexityAgent:
    """Classifies query complexity (Easy/Medium/Hard) for routing"""
    
    def __init__(self, finetuning_filename='finetuning_corpus_detailed.json', 
                 diffusion_filename='diffusion_corpus_detailed.json'):
        print("[*] Initializing ComplexityAgent...")
        
        # Load corpora
        self.finetuning_filename = finetuning_filename
        self.diffusion_filename = diffusion_filename
        
        # Define complexity patterns
        self.easy_keywords = [
            "scan port", "check port", "list services", "basic scan",
            "host discovery", "ping", "which ports", "open port",
            "is open", "listening", "service version", "simple",
            "default", "standard", "quick", "fast"
        ]
        
        self.medium_keywords = [
            "stealth", "timing", "firewall", "evasion", "scripts",
            "vuln", "version detection", "aggressive", "scan types",
            "service detection", "version scan", "script scan",
            "safe scripts", "default scripts", "ssl", "certificate"
        ]
        
        self.hard_keywords = [
            "os detection", "all", "comprehensive", "complete",
            "vulnerability", "exploit", "brute force", "crack",
            "authentication", "sensitive", "advanced", "custom",
            "complex", "difficult", "reconnaissance"
        ]
        
        print("[✓] ComplexityAgent ready")
    
    def classify(self, query: str) -> Dict[str, any]:
        """Classify query complexity"""
        query_lower = query.lower()
        
        # Count keyword matches
        easy_score = sum(1 for kw in self.easy_keywords if kw in query_lower)
        medium_score = sum(1 for kw in self.medium_keywords if kw in query_lower)
        hard_score = sum(1 for kw in self.hard_keywords if kw in query_lower)
        
        # Determine complexity level
        scores = {'Easy': easy_score, 'Medium': medium_score, 'Hard': hard_score}
        max_score = max(scores.values()) if scores else 0
        
        if max_score == 0:
            level = 'Medium'  # Default
            confidence = 0.5
        else:
            level = [k for k, v in scores.items() if v == max_score][0]
            confidence = max_score / (easy_score + medium_score + hard_score + 1)
        
        # Reason
        if level == 'Easy':
            reason = f"Query contains basic scanning keywords: {easy_score} matches"
        elif level == 'Medium':
            reason = f"Query contains intermediate keywords: {medium_score} matches"
        else:
            reason = f"Query contains advanced keywords: {hard_score} matches"
        
        return {
            'level': level,
            'confidence': min(confidence, 1.0),
            'reason': reason
        }

# ============= FASTAPI APP =============

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