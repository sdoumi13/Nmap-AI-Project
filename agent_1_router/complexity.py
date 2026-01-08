"""
complexity.py - 
Better logic for Easy/Medium queries, prevents SLM hallucination bias
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Literal, List, Tuple, Any
import sys
import os
from pathlib import Path
import json
from collections import defaultdict
import re
import httpx
import asyncio

# Colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Setup imports
current_dir = Path(__file__).resolve().parent
project_root = Path(current_dir).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class SLMClassifier:
    """SLM-based complexity classifier using Qwen2.5-Coder-3B"""
    
    def __init__(self, api_url: str = "http://192.168.11.1:1234/v1/chat/completions"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.model_name = "qwen2.5-coder-3b-instruct"
    
    async def classify(self, user_query: str, corpus_examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """Classify query complexity using SLM"""
        
        prompt = f"""You are an expert in Nmap. Classify ONLY based on what the user actually requests.

**Rules:**
- EASY: Simple single-purpose scans (1-2 basic flags)
  Examples: "scan ports", "ping hosts", "check if up", "discover network"
  
- MEDIUM: 3 techniques or moderate complexity
  Examples: "scan with OS detection", "get service versions", "enumerate services"
  
- HARD: 4+ techniques OR advanced evasion/scripting
  Examples: "comprehensive fingerprinting with OS+version+banner+scripts", "stealth scan with decoys", "NSE vulnerability scanning"

**User Query:** "{user_query}"

**IMPORTANT:** Count ONLY what the user explicitly requests. Don't assume techniques not mentioned.

Examples:
- "Check if host is up" = 1 technique (ping) = EASY
- "Scan ports with OS detection" = 2 techniques = EASY/MEDIUM
- "Comprehensive fingerprinting with OS, version, banner, headers" = 4+ techniques = HARD

JSON response (no markdown):
{{
    "complexity": "Easy",
    "confidence": 0.9,
    "reasoning": "Single technique: ping/availability check",
    "technique_count": 1
}}"""

        try:
            response = await self.client.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are precise. Count ONLY explicitly mentioned techniques. Don't hallucinate. Respond in valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.05,  # Very low for consistency
                    "max_tokens": 250
                }
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Clean JSON
            content = content.strip()
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\n?', '', content)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(content)
            
            # Normalize complexity value
            complexity = parsed.get('complexity', 'Medium')
            if complexity not in ['Easy', 'Medium', 'Hard']:
                complexity = 'Medium'
            
            return {
                "complexity": complexity,
                "confidence": parsed.get('confidence', 0.5),
                "reasoning": parsed.get('reasoning', 'SLM classification'),
                "technique_count": parsed.get('technique_count', 0)
            }
        
        except httpx.ConnectError:
            print(f"{RED}[SLM] Connection error: LM Studio not running on port 1234?{RESET}")
            return {
                "complexity": "Medium",
                "confidence": 0.3,
                "reasoning": "SLM unavailable - using fallback"
            }
        except Exception as e:
            print(f"{RED}[SLM Classifier] Error: {str(e)[:100]}{RESET}")
            return {
                "complexity": "Medium",
                "confidence": 0.3,
                "reasoning": f"SLM error: {str(e)[:50]}"
            }
    
    def _format_examples(self, examples: List[str]) -> str:
        """Format examples for prompt"""
        if not examples:
            return "  (No examples available)"
        return "\n".join([f"  - {ex}" for ex in examples])
    
    async def health_check(self) -> bool:
        """Check if SLM is available"""
        try:
            response = await self.client.get(
                self.api_url.replace('/v1/chat/completions', '/v1/models'),
                timeout=5.0
            )
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        await self.client.aclose()


class ComplexityAgent:
    """Hybrid Complexity Classifier with improved logic"""
    
    def __init__(self, 
                 rag_filename='rag_corpus_detailed.json',
                 diffusion_filename='diffusion_corpus_detailed.json',
                 finetuning_filename='finetuning_corpus_detailed.json'):
        
        print(f"{CYAN}[*] Initializing Enhanced ComplexityAgent v3...{RESET}")
        
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
        
        # Initialize SLM classifier
        print(f"{YELLOW}[*] Connecting to Qwen2.5-Coder-3B (Port 1234)...{RESET}")
        self.slm_classifier = SLMClassifier(api_url="http://192.168.11.1:1234/v1/chat/completions")
        
        print(f"{GREEN}[✓] Loaded {len(self.easy_examples)} EASY, {len(self.medium_examples)} MEDIUM, {len(self.hard_examples)} HARD examples{RESET}")
        print(f"{GREEN}[✓] ComplexityAgent ready{RESET}")
    
    def _load_corpus(self, filename: str) -> Dict:
        """Load corpus JSON file"""
        filepath = self.datasets_path / filename
        if not filepath.exists():
            print(f"{YELLOW}[!] Warning: {filename} not found at {filepath}{RESET}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"{RED}[!] Error loading {filename}: {e}{RESET}")
            return {}
    
    def _index_corpus(self):
        """Index corpus examples by difficulty level"""
        
        # Index RAG corpus
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
        
        # Index Diffusion corpus
        if self.diffusion_corpus and 'training_data' in self.diffusion_corpus:
            for entry in self.diffusion_corpus['training_data']:
                complexity = entry.get('complexity_level', 5)
                description = entry.get('text_description', '')
                command = entry.get('target_command', '')
                
                example = {
                    'intent': description,
                    'command': command,
                    'context': description,
                    'source': 'diffusion'
                }
                
                if complexity <= 3:
                    self.easy_examples.append(example)
                elif complexity <= 6:
                    self.medium_examples.append(example)
                else:
                    self.hard_examples.append(example)
        
        # Index Finetuning corpus
        if self.finetuning_corpus and 'conversations' in self.finetuning_corpus:
            for conv in self.finetuning_corpus['conversations']:
                difficulty = conv.get('difficulty', 'medium').lower()
                
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
        words = re.findall(r'\w+', text)
        return set(words)
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_best_matches(self, query: str, examples: List[Dict], top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find best matching examples"""
        query_tokens = self._tokenize(query)
        
        similarities = []
        for example in examples:
            intent = example.get('intent', '')
            context = example.get('context', '')
            combined = f"{intent} {context}"
            
            tokens = self._tokenize(combined)
            similarity = self._jaccard_similarity(query_tokens, tokens)
            similarities.append((example, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _keyword_classification(self, query: str) -> Tuple[str, float, int]:
        """Enhanced keyword-based heuristic classification"""
        query_lower = query.lower()
        
        # Easy indicators (1-2 simple operations)
        simple_keywords = {
            'scan': 1, 'ping': 1, 'port': 1, 'check': 1, 'simple': 1, 
            'basic': 1, 'discover': 1, 'host': 1, 'up': 1, 'quick': 1, 'fast': 1
        }
        
        # Medium indicators (3-4 techniques)
        medium_keywords = {
            'version': 1, 'service': 1, 'fingerprint': 1, 'enumerate': 1, 
            'banner': 1, 'detect': 1
        }
        
        # Hard indicators (5+ or advanced techniques)
        complex_keywords = {
            'stealth': 2, 'evade': 2, 'bypass': 2, 'ids': 2, 'firewall': 2,
            'fragment': 2, 'decoy': 2, 'obfuscate': 2, 'spoof': 2, 
            'brute': 2, 'exploit': 2, 'vulnerability': 2, 'nse': 1,
            'comprehensive': 2, 'complet': 2, 'maximum': 2, 'maximale': 2,
            'système': 1, 'énumération': 1
        }
        
        # Count explicit techniques mentioned
        technique_count = 0
        complexity_score = 0
        
        # Check for explicit technique mentions
        if any(w in query_lower for w in ['os', 'operating system', 'système']):
            technique_count += 1
            complexity_score += 1
        
        if any(w in query_lower for w in ['version', 'service']):
            technique_count += 1
            complexity_score += 1
        
        if any(w in query_lower for w in ['banner', 'bannière']):
            technique_count += 1
            complexity_score += 1
        
        if any(w in query_lower for w in ['script', 'nse']):
            technique_count += 1
            complexity_score += 2
        
        if any(w in query_lower for w in ['header', 'en-tête', 'http']):
            technique_count += 1
            complexity_score += 1
        
        # Check keyword categories
        simple_count = sum(1 for kw in simple_keywords if kw in query_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in query_lower)
        complex_count = sum(complex_keywords.get(kw, 0) for kw in complex_keywords if kw in query_lower)
        
        complexity_score += complex_count
        
        # Decision logic with clear thresholds
        if technique_count >= 5 or complexity_score >= 8:
            return ('Hard', 0.85, technique_count)
        elif technique_count >= 3 or complexity_score >= 4:
            return ('Medium', 0.75, technique_count)
        elif simple_count > 0 and technique_count <= 1 and complexity_score <= 2:
            return ('Easy', 0.80, max(technique_count, 1))
        else:
            return ('Medium', 0.60, technique_count)
    
    async def classify(self, query: str) -> Dict[str, any]:
        """Hybrid classification with improved weighting and sanity checks"""
        
        print(f"\n{CYAN}[Complexity] Classifying: '{query}'{RESET}")
        
        # Layer 1: Corpus-based Similarity
        easy_matches = self._find_best_matches(query, self.easy_examples, top_k=5)
        medium_matches = self._find_best_matches(query, self.medium_examples, top_k=5)
        hard_matches = self._find_best_matches(query, self.hard_examples, top_k=5)
        
        easy_score = sum(s for _, s in easy_matches) / len(easy_matches) if easy_matches else 0.0
        medium_score = sum(s for _, s in medium_matches) / len(medium_matches) if medium_matches else 0.0
        hard_score = sum(s for _, s in hard_matches) / len(hard_matches) if hard_matches else 0.0
        
        print(f"  [Corpus] Easy: {easy_score:.3f} | Medium: {medium_score:.3f} | Hard: {hard_score:.3f}")
        
        # Layer 2: Keyword Heuristics (ground truth)
        keyword_level, keyword_confidence, technique_count = self._keyword_classification(query)
        print(f"  [Keywords] Level: {keyword_level} | Confidence: {keyword_confidence:.3f} | Techniques: {technique_count}")
        
        # Layer 3: SLM Classification
        corpus_examples = {
            'easy': [ex['intent'] for ex in self.easy_examples[:5]],
            'medium': [ex['intent'] for ex in self.medium_examples[:5]],
            'hard': [ex['intent'] for ex in self.hard_examples[:5]]
        }
        
        slm_result = await self.slm_classifier.classify(query, corpus_examples)
        
        slm_level = slm_result['complexity']
        slm_confidence = slm_result['confidence']
        slm_reasoning = slm_result['reasoning']
        slm_technique_count = slm_result.get('technique_count', 0)
        
        print(f"  [SLM] Level: {slm_level} | Confidence: {slm_confidence:.3f} | Techniques: {slm_technique_count}")
        print(f"        Reasoning: {slm_reasoning[:100]}")
        
        # SANITY CHECK: Detect SLM hallucination
        hallucination_detected = False
        if slm_technique_count > technique_count + 2:  # SLM claims 2+ more techniques
            print(f"  {YELLOW}[WARNING] SLM hallucination detected: claims {slm_technique_count} techniques, actual: {technique_count}{RESET}")
            hallucination_detected = True
            slm_confidence *= 0.4  # Severely reduce SLM confidence
        
        # Determine corpus best match
        corpus_best = max([('Easy', easy_score), ('Medium', medium_score), ('Hard', hard_score)], key=lambda x: x[1])
        corpus_level, corpus_conf = corpus_best
        
        # IMPROVED WEIGHTING LOGIC
        votes = {'Easy': 0.0, 'Medium': 0.0, 'Hard': 0.0}
        
        # Strong agreement between corpus and keywords = high trust
        if corpus_level == keyword_level and corpus_conf > 0.3:
            # Keywords + Corpus agree = strong signal
            votes[keyword_level] += 0.45 * keyword_confidence
            votes[corpus_level] += 0.35 * corpus_conf
            votes[slm_level] += 0.20 * slm_confidence  # Lower SLM weight
            
            print(f"  [AGREEMENT] Corpus and Keywords agree on {keyword_level}")
        else:
            # Normal weighting
            if hallucination_detected:
                # Don't trust SLM
                votes[keyword_level] += 0.60 * keyword_confidence
                votes[corpus_level] += 0.30 * corpus_conf
                votes[slm_level] += 0.10 * slm_confidence
            else:
                votes[keyword_level] += 0.40 * keyword_confidence
                votes[corpus_level] += 0.30 * corpus_conf
                votes[slm_level] += 0.30 * slm_confidence
        
        # Apply technique count boost for Hard classification
        if technique_count >= 5:
            votes['Hard'] += 0.25
            print(f"  [BOOST] Hard +0.25 for {technique_count} techniques")
        
        # Final decision
        final_level = max(votes, key=votes.get)
        final_confidence = votes[final_level]
        
        # Override if very strong consensus (all 3 layers agree)
        if not hallucination_detected and slm_level == keyword_level == corpus_level:
            final_level = slm_level
            final_confidence = min(0.95, final_confidence + 0.15)
            print(f"  [CONSENSUS] All layers agree on {final_level}")
        
        # Use keyword reasoning if SLM is hallucinating
        if hallucination_detected:
            reasoning = f"Keywords detected {technique_count} techniques: {keyword_level}"
        else:
            reasoning = slm_reasoning if slm_confidence > 0.6 else f"Keywords: {keyword_level} ({technique_count} techniques)"
        
        if final_level == 'Easy':
            matched = easy_matches
        elif final_level == 'Medium':
            matched = medium_matches
        else:
            matched = hard_matches
        
        top_match = matched[0][0].get('intent', '') if matched else ''
        
        print(f"  [FINAL] Level: {final_level} | Confidence: {final_confidence:.3f}")
        
        return {
            'level': final_level,
            'confidence': final_confidence,
            'reason': reasoning,
            'scores': {'easy': easy_score, 'medium': medium_score, 'hard': hard_score},
            'top_match': top_match,
            'technique_count': technique_count,  # Use keyword count, not SLM
            'layers': {
                'corpus': (corpus_level, corpus_conf),
                'keywords': (keyword_level, keyword_confidence),
                'slm': (slm_level, slm_confidence),
                'hallucination': hallucination_detected
            }
        }
    
    async def close(self):
        await self.slm_classifier.close()


# FastAPI Application
app = FastAPI(title="Enhanced Nmap Complexity Classifier v3", version="2.2")

class ComplexityRequest(BaseModel):
    query: str
    user_id: str = "anonymous"

class ComplexityResponse(BaseModel):
    query: str
    complexity: Literal["EASY", "MEDIUM", "HARD"]
    confidence: float
    recommended_agent: Literal["RAG", "DIFFUSION"]
    reasoning: str
    technique_count: int = 0

complexity_agent = None

@app.on_event("startup")
async def startup():
    global complexity_agent
    print(f"{CYAN}🚀 Loading Enhanced ComplexityAgent v3...{RESET}")
    complexity_agent = ComplexityAgent()
    
    # Health check
    healthy = await complexity_agent.slm_classifier.health_check()
    print(f"SLM Health: {'🟢 ONLINE' if healthy else '🔴 OFFLINE'}")
    print(f"{GREEN}✅ ComplexityAgent ready{RESET}")

@app.get("/")
def root():
    return {"service": "Enhanced Nmap Complexity Classifier", "version": "2.2"}

@app.get("/health")
def health():
    return {"status": "healthy" if complexity_agent else "initializing"}

@app.post("/classify", response_model=ComplexityResponse)
async def classify_query(request: ComplexityRequest):
    if not complexity_agent:
        raise HTTPException(status_code=503, detail="Classifier not ready")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await complexity_agent.classify(request.query)
        
        level_map = {"Easy": "EASY", "Medium": "MEDIUM", "Hard": "HARD"}
        agent_map = {"Easy": "RAG", "Medium": "DIFFUSION", "Hard": "DIFFUSION"}
        
        complexity = level_map.get(result['level'], "MEDIUM")
        
        return ComplexityResponse(
            query=request.query,
            complexity=complexity,
            confidence=result['confidence'],
            recommended_agent=agent_map.get(result['level'], "DIFFUSION"),
            reasoning=result['reason'],
            technique_count=result.get('technique_count', 0)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.on_event("shutdown")
async def shutdown():
    if complexity_agent:
        await complexity_agent.close()


if __name__ == "__main__":
    import uvicorn
    print("""
    ╔═══════════════════════════════════════════╗
    ║   ENHANCED COMPLEXITY CLASSIFIER v3       ║
    ║   WITH HALLUCINATION DETECTION            ║
    ╚═══════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="info")