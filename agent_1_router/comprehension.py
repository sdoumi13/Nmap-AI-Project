"""
Enhanced Comprehension Agent with SLM Integration
Qwen2.5-Coder-3B on Port 1234
"""

import numpy as np
import json
import os
import httpx
import asyncio
import re
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class SLMClient:
    """Client for Qwen2.5-Coder-3B (Port 1234)"""
    
    def __init__(self, api_url: str = "http://192.168.11.1:1234/v1/chat/completions"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.model_name = "qwen2.5-coder-3b-instruct"
    
    async def analyze_intent(self, user_query: str, nmap_examples: List[str]) -> Dict[str, Any]:
        """Ask SLM to determine if query is Nmap-related"""
        
        examples_text = "\n".join([f"- {ex}" for ex in nmap_examples[:5]])
        
        # FIX: Add explicit Nmap keywords to help SLM
        prompt = f"""You are an expert in network security and Nmap scanning.

**Nmap-related topics include:**
- Port scanning (SYN, TCP, UDP, stealth)
- Service/version detection (-sV)
- OS fingerprinting (-O)
- Vulnerability scanning (--script vuln)
- Network discovery (ping scans)
- Firewall/IDS evasion (decoys, fragmentation, timing)
- NSE scripts (--script)
- Target specification (IP, CIDR, domain)

**Example Nmap queries:**
{examples_text}

**User Query:** "{user_query}"

**Task:** Determine if this query is about Nmap network scanning.

**IMPORTANT:** Queries about "IDS bypass", "firewall evasion", "decoys", "stealth scan", "fragmentation" ARE Nmap-related because these are common Nmap techniques.

Respond ONLY with valid JSON (no markdown):
{{
    "is_nmap_related": true,
    "confidence": 0.95,
    "reasoning": "This query asks about X, which is a Nmap technique",
    "extracted_intent": "brief intent description"
}}"""

        try:
            response = await self.client.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a cybersecurity expert specializing in Nmap. Always respond in valid JSON without markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 250
                }
            )
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            
            # Clean JSON (remove markdown if present)
            content = content.strip()
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\n?', '', content)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(content)
            
            # FIX: Validate confidence makes sense
            if parsed.get('is_nmap_related'):
                # If marked as related, confidence should be positive
                if parsed.get('confidence', 0) < 0.5:
                    print(f"  {YELLOW}[SLM] Warning: Low confidence for positive result, boosting to 0.7{RESET}")
                    parsed['confidence'] = 0.7
            
            return parsed
            
        except httpx.ConnectError:
            print(f"{RED}[SLM] Connection error: LM Studio not running on port 1234?{RESET}")
            return {
                "is_nmap_related": False,
                "confidence": 0.0,
                "error": True,
                "reasoning": "LM Studio connection failed",
                "extracted_intent": ""
            }
        except Exception as e:
            print(f"{RED}[SLM] Error: {str(e)[:100]}{RESET}")
            return {
                "is_nmap_related": False,
                "confidence": 0.0,
                "error": True,
                "reasoning": f"SLM error: {str(e)[:50]}",
                "extracted_intent": ""
            }
    
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


class ComprehensionAgent:
    """Hybrid Comprehension Agent (TF-IDF + SBERT + SLM)"""
    
    def __init__(self, corpus_filename='rag_corpus_detailed.json'):
        print(f"{CYAN}🧠 Initializing Enhanced Comprehension Agent...{RESET}")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.corpus_path = os.path.join(project_root, 'datasets', corpus_filename)
        
        self.load_rag_corpus()
        
        self.noise_corpus = [
            "what is the weather", "bake a cake", "who won the world cup", 
            "play music", "center a div", "install python", "fix wifi"
        ]
        
        # FIX: Add Nmap-specific keywords for better detection
        self.nmap_keywords = {
            "scan", "nmap", "port", "ip", "network", "host", "service",
            "stealth", "syn", "tcp", "udp", "version", "detect",
            "firewall", "ids", "bypass", "evasion", "decoy", "fragment",
            "vulnerability", "vuln", "exploit", "script", "nse",
            "os", "fingerprint", "discovery", "ping", "traceroute"
        }
        
        self.all_corpus = self.nmap_corpus + self.noise_corpus
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_corpus)
        self.n_nmap = len(self.nmap_corpus)
        
        print(f"   {YELLOW}Loading SBERT model...{RESET}")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reference_embedding = self.embedding_model.encode(
            "scan network ports service detection vulnerability assessment nse host discovery stealth firewall evasion ids bypass decoy fragmentation",
            convert_to_tensor=True
        )
        
        print(f"   {YELLOW}Connecting to Qwen2.5-Coder-3B (Port 1234)...{RESET}")
        self.slm_client = SLMClient()
        
        print(f"   {GREEN}✅ Enhanced Comprehension Agent ready!{RESET}")
    
    def load_rag_corpus(self):
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                rag_data = json.load(f)
            
            documents = rag_data.get('knowledge_base', rag_data.get('documents', []))
            self.nmap_corpus = []
            self.corpus_metadata = []
            
            for doc in documents:
                self.nmap_corpus.append(f"{doc.get('intent', '')} {doc.get('explanation', '')}")
                self.corpus_metadata.append({'id': doc.get('id', 'unknown'), 'category': doc.get('category', 'general')})
        except Exception as e:
            print(f"{RED}Error loading corpus: {e}{RESET}")
            self.nmap_corpus = ["scan ports"]
            self.corpus_metadata = [{}]

    async def analyze(self, user_query: str) -> Dict[str, Any]:
        print(f"\n{CYAN}[Comprehension] Analyzing: '{user_query}'{RESET}")
        
        # FIX: Keyword boost detection first
        query_lower = user_query.lower()
        keyword_matches = sum(1 for kw in self.nmap_keywords if kw in query_lower)
        keyword_boost = min(keyword_matches * 0.1, 0.3)  # Max 0.3 boost
        
        if keyword_boost > 0:
            print(f"  [KEYWORDS] Matched {keyword_matches} Nmap keywords → boost: +{keyword_boost:.2f}")
        
        # Layer 1: TF-IDF
        query_vec = self.tfidf_vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        max_nmap_tfidf = np.max(scores[:self.n_nmap]) if self.n_nmap > 0 else 0
        max_noise_tfidf = np.max(scores[self.n_nmap:]) if len(scores) > self.n_nmap else 0
        
        print(f"  [TF-IDF] Nmap: {max_nmap_tfidf:.3f} | Noise: {max_noise_tfidf:.3f}")
        
        # Layer 2: SBERT
        query_emb = self.embedding_model.encode(user_query, convert_to_tensor=True)
        from sentence_transformers import util
        semantic_score = float(util.cos_sim(query_emb, self.reference_embedding)[0][0])
        
        print(f"  [SBERT] Semantic: {semantic_score:.3f}")
        
        # Layer 3: SLM (with graceful degradation)
        slm_result = await self.slm_client.analyze_intent(user_query, self.nmap_corpus[:5])
        
        slm_weight = 0.35  # Reduced from 0.4
        slm_score = 0
        
        if slm_result.get('error'):
            print(f"  {YELLOW}[WARNING] SLM Offline - Using fallback scoring{RESET}")
            base_score = (max_nmap_tfidf * 0.5) + (semantic_score * 0.5)
        else:
            # FIX: Correct handling of SLM confidence
            is_related = slm_result.get('is_nmap_related', False)
            confidence = slm_result.get('confidence', 0)
            
            # If SLM says it's related with high confidence, use positive score
            if is_related and confidence >= 0.6:
                slm_score = confidence * slm_weight
            elif is_related:
                # Related but low confidence
                slm_score = 0.5 * slm_weight
            else:
                # Not related - apply penalty only if confidence is high
                if confidence >= 0.7:
                    slm_score = -0.1 * slm_weight
                else:
                    slm_score = 0  # Uncertain, no penalty
            
            base_score = (max_nmap_tfidf * 0.35) + (semantic_score * 0.3)
            
            print(f"  [SLM] Related: {is_related} | Conf: {confidence:.3f} | Score: {slm_score:+.3f}")

        # FIX: Apply keyword boost BEFORE calculating final score
        final_score = base_score + slm_score + keyword_boost

        # Additional boost for strong Nmap indicators
        if max_nmap_tfidf > 0.4:
            final_score += 0.1
            print(f"  [BOOST] Strong TF-IDF match → +0.10")
        
        # Penalty only if noise is significantly higher
        if max_noise_tfidf > max_nmap_tfidf + 0.2:
            final_score -= 0.15
            print(f"  [PENALTY] Noise dominant → -0.15")

        # FIX: Lower threshold for better sensitivity
        THRESHOLD = 0.22  # Was 0.25
        is_relevant = final_score >= THRESHOLD
        
        print(f"  [FINAL] Score: {final_score:.3f} | Threshold: {THRESHOLD} | Relevant: {is_relevant}")
        
        return {
            "relevant": is_relevant,
            "score": final_score,
            "reason": slm_result.get('reasoning', "Fallback scoring applied"),
            "extracted_intent": slm_result.get('extracted_intent', ''),
            "layers": {
                "tfidf": max_nmap_tfidf,
                "sbert": semantic_score,
                "slm": slm_score,
                "keyword_boost": keyword_boost
            }
        }

    async def close(self):
        await self.slm_client.close()


async def test_comprehension():
    agent = ComprehensionAgent()
    
    # Health check
    healthy = await agent.slm_client.health_check()
    print(f"\n{'='*60}")
    print(f"SLM Health: {'🟢 ONLINE' if healthy else '🔴 OFFLINE'}")
    print(f"{'='*60}\n")
    
    queries = [
        "scan 192.168.1.1",
        "Bypass IDS firewall using decoys",
        "stealth scan with fragmentation",
        "detect OS version",
        "make a cake",
        "what is the weather"
    ]
    
    for q in queries:
        res = await agent.analyze(q)
        print(f"\nQuery: '{q}'")
        print(f"Result: {res['relevant']} (score: {res['score']:.2f})")
        if res['relevant']:
            print(f"Intent: {res['extracted_intent']}")
        print("-"*60)
    
    await agent.close()

if __name__ == "__main__":
    asyncio.run(test_comprehension())