"""
Agent 1 - Complexity Classifier REST API
Classifies query complexity: EASY → RAG | MEDIUM/HARD → Diffusion
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Literal
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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


# ============= CLASSIFIER =============

class ComplexityClassifier:
    """
    Classifie la complexité d'une requête Nmap
    
    EASY: Simple scans (web ports, basic TCP)
    MEDIUM: Multi-flag scans (OS detection, version)
    HARD: Advanced scenarios (stealth + scripts + timing)
    """
    
    def __init__(self):
        # Keywords pour chaque niveau
        self.easy_keywords = [
            'scan', 'check', 'web', 'port 80', 'port 443', 
            'http', 'https', 'simple', 'basic', 'quick'
        ]
        
        self.medium_keywords = [
            'version', 'os detection', 'service', 'fingerprint',
            'multiple ports', 'range', 'subnet', 'network'
        ]
        
        self.hard_keywords = [
            'stealth', 'evade', 'bypass', 'firewall', 'ids',
            'script', 'vulnerability', 'exploit', 'advanced',
            'timing', 'decoy', 'fragmentation', 'ipv6'
        ]
        
        # Patterns de complexité
        self.complexity_patterns = {
            'multi_flag': r'-[a-zA-Z]{2,}',  # -sS, -sV, -O ensemble
            'port_range': r'\d+-\d+|all ports',
            'script_usage': r'script|nse|vuln',
            'timing': r'timing|T[0-5]|slow|fast',
            'cidr': r'/\d{1,2}',  # Subnet mask
        }
    
    def classify(self, query: str) -> Dict:
        """
        Classifie la requête et recommande l'agent approprié
        
        Returns:
            {
                'complexity': 'EASY'|'MEDIUM'|'HARD',
                'confidence': float,
                'recommended_agent': 'RAG'|'DIFFUSION',
                'reasoning': str
            }
        """
        query_lower = query.lower()
        
        # Score chaque catégorie
        easy_score = self._count_keywords(query_lower, self.easy_keywords)
        medium_score = self._count_keywords(query_lower, self.medium_keywords)
        hard_score = self._count_keywords(query_lower, self.hard_keywords)
        
        # Détecter patterns complexes
        pattern_count = sum(
            1 for pattern in self.complexity_patterns.values()
            if re.search(pattern, query_lower)
        )
        
        # Ajuster les scores avec patterns
        if pattern_count >= 3:
            hard_score += 2
        elif pattern_count >= 2:
            medium_score += 1
        
        # Longueur de la requête (indicateur de complexité)
        word_count = len(query.split())
        if word_count > 15:
            hard_score += 1
        elif word_count > 10:
            medium_score += 1
        
        # Décision finale
        scores = {
            'EASY': easy_score,
            'MEDIUM': medium_score,
            'HARD': hard_score
        }
        
        complexity = max(scores, key=scores.get)
        max_score = scores[complexity]
        total_score = sum(scores.values()) or 1  # Éviter division par zéro
        
        confidence = max_score / total_score
        
        # Recommandation d'agent
        if complexity == 'EASY':
            agent = 'RAG'
            reasoning = "Simple query with basic scan requirements. RAG retrieval is sufficient."
        elif complexity == 'MEDIUM':
            agent = 'DIFFUSION'
            reasoning = "Moderate complexity with multiple flags. Diffusion model recommended."
        else:  # HARD
            agent = 'DIFFUSION'
            reasoning = "Complex scenario with advanced requirements. Diffusion iterative generation needed."
        
        return {
            'complexity': complexity,
            'confidence': confidence,
            'recommended_agent': agent,
            'reasoning': reasoning,
            'scores': scores  # Pour debugging
        }
    
    def _count_keywords(self, text: str, keywords: list) -> int:
        """Compte les occurrences de mots-clés"""
        return sum(1 for kw in keywords if kw in text)


# ============= API ENDPOINTS =============

classifier = ComplexityClassifier()


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
    return {
        "status": "healthy",
        "classifier": "ready"
    }


@app.post("/classify", response_model=ComplexityResponse)
def classify_query(request: ComplexityRequest):
    """
    Classifie la complexité d'une requête Nmap
    
    Returns:
        - complexity: EASY | MEDIUM | HARD
        - confidence: 0.0 - 1.0
        - recommended_agent: RAG | DIFFUSION
        - reasoning: Explication de la décision
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = classifier.classify(request.query)
        
        return ComplexityResponse(
            query=request.query,
            complexity=result['complexity'],
            confidence=result['confidence'],
            recommended_agent=result['recommended_agent'],
            reasoning=result['reasoning']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


# ============= MAIN =============

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════════╗
    ║   NMAP COMPLEXITY CLASSIFIER API          ║
    ║   Easy → RAG | Medium/Hard → Diffusion    ║
    ╚════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7000,
        log_level="info"
    )