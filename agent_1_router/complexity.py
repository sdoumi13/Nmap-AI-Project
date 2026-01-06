"""
complexity_api.py
API REST utilisant ComplexityAgent local basé sur les corpus
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Literal, List, Tuple
import sys
import os
from pathlib import Path
import json
from collections import defaultdict
import re

# Setup imports
current_dir = Path(__file__).resolve().parent
project_root = Path(current_dir).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============= COMPLEXITY AGENT CLASS =============

class ComplexityAgent:
    """
    Classifies query complexity (Easy/Medium/Hard) based on corpus similarity
    Uses 3 datasets: RAG, Diffusion, and Finetuning corpus
    """
    
    def __init__(self, 
                 rag_filename='rag_corpus_detailed.json',
                 diffusion_filename='diffusion_corpus_detailed.json',
                 finetuning_filename='finetuning_corpus_detailed.json'):
        print("[*] Initializing ComplexityAgent with corpus-based approach...")
        
        self.datasets_path = Path(project_root) / "datasets"
        
        # Load corpora
        self.rag_corpus = self._load_corpus(rag_filename)
        self.diffusion_corpus = self._load_corpus(diffusion_filename)
        self.finetuning_corpus = self._load_corpus(finetuning_filename)
        
        # Index corpus by difficulty
        self.easy_examples = []
        self.medium_examples = []
        self.hard_examples = []
        
        self._index_corpus()
        
        print(f"[✓] Loaded {len(self.easy_examples)} EASY, {len(self.medium_examples)} MEDIUM, {len(self.hard_examples)} HARD examples")
        print("[✓] ComplexityAgent ready")
    
    def _load_corpus(self, filename: str) -> Dict:
        """Load corpus JSON file"""
        filepath = self.datasets_path / filename
        if not filepath.exists():
            print(f"[!] Warning: {filename} not found at {filepath}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Error loading {filename}: {e}")
            return {}
    
    def _index_corpus(self):
        """Index corpus examples by difficulty level"""
        
        # Index RAG corpus (easiest - direct examples)
        if self.rag_corpus and 'knowledge_base' in self.rag_corpus:
            for entry in self.rag_corpus['knowledge_base']:
                difficulty = entry.get('difficulty', 'medium').lower()
                intent = entry.get('intent', '')
                command = entry.get('command', '')
                context = entry.get('context', '')
                
                example = {
                    'intent': intent,
                    'command': command,
                    'context': context,
                    'source': 'rag'
                }
                
                if difficulty == 'easy':
                    self.easy_examples.append(example)
                elif difficulty == 'medium':
                    self.medium_examples.append(example)
                else:
                    self.hard_examples.append(example)
        
        # Index Diffusion corpus (mixed complexities)
        if self.diffusion_corpus and 'training_data' in self.diffusion_corpus:
            for entry in self.diffusion_corpus['training_data']:
                complexity = entry.get('complexity_level', 2)
                description = entry.get('text_description', '')
                command = entry.get('target_command', '')
                
                example = {
                    'intent': description,
                    'command': command,
                    'context': description,
                    'source': 'diffusion'
                }
                
                if complexity <= 1:
                    self.easy_examples.append(example)
                elif complexity <= 2:
                    self.medium_examples.append(example)
                else:
                    self.hard_examples.append(example)
        
        # Index Finetuning corpus
        if self.finetuning_corpus and 'conversations' in self.finetuning_corpus:
            for conv in self.finetuning_corpus['conversations']:
                difficulty = conv.get('difficulty', 'medium').lower()
                
                # Extract user query and command from conversation
                turns = conv.get('turns', [])
                user_query = ""
                for turn in turns:
                    if turn.get('role') == 'user':
                        user_query = turn.get('content', '')
                        break
                
                if user_query:
                    example = {
                        'intent': user_query,
                        'command': '',
                        'context': user_query,
                        'source': 'finetuning'
                    }
                    
                    if difficulty == 'easy':
                        self.easy_examples.append(example)
                    elif difficulty == 'medium':
                        self.medium_examples.append(example)
                    else:
                        self.hard_examples.append(example)
    
    
    def _tokenize(self, text: str) -> set:
        """Simple tokenization"""
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\w+', text)
        return set(words)
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_best_matches(self, query: str, examples: List[Dict], top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Find best matching examples from a corpus"""
        query_tokens = self._tokenize(query)
        
        similarities = []
        for example in examples:
            intent = example.get('intent', '')
            intent_tokens = self._tokenize(intent)
            
            similarity = self._jaccard_similarity(query_tokens, intent_tokens)
            similarities.append((example, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def classify(self, query: str) -> Dict[str, any]:
        """
        Classify query complexity using corpus-based similarity
        
        Returns:
            {
                'level': 'Easy'|'Medium'|'Hard',
                'confidence': float (0-1),
                'reason': str,
                'matched_examples': list of matched corpus examples
            }
        """
        
        # Find best matches in each difficulty level
        easy_matches = self._find_best_matches(query, self.easy_examples, top_k=3)
        medium_matches = self._find_best_matches(query, self.medium_examples, top_k=3)
        hard_matches = self._find_best_matches(query, self.hard_examples, top_k=3)
        
        # Calculate average similarity for each level
        easy_score = sum(s for _, s in easy_matches) / len(easy_matches) if easy_matches else 0.0
        medium_score = sum(s for _, s in medium_matches) / len(medium_matches) if medium_matches else 0.0
        hard_score = sum(s for _, s in hard_matches) / len(hard_matches) if hard_matches else 0.0
        
        # Determine level based on highest score
        scores = {
            'Easy': easy_score,
            'Medium': medium_score,
            'Hard': hard_score
        }
        
        level = max(scores, key=scores.get)
        confidence = scores[level]
        
        # Build detailed reason
        if level == 'Easy':
            matched = easy_matches
            reason = f"Query similar to EASY examples (score: {easy_score:.2f})"
        elif level == 'Medium':
            matched = medium_matches
            reason = f"Query similar to MEDIUM examples (score: {medium_score:.2f})"
        else:
            matched = hard_matches
            reason = f"Query similar to HARD examples (score: {hard_score:.2f})"
        
        # Add top matched example
        if matched:
            top_match = matched[0]
            example_intent = top_match[0].get('intent', '')[:50]
            reason += f" | Matched: '{example_intent}...'"
        
        return {
            'level': level,
            'confidence': confidence,
            'reason': reason,
            'scores': {
                'easy': easy_score,
                'medium': medium_score,
                'hard': hard_score
            },
            'top_match': matched[0][0].get('intent', '') if matched else ''
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