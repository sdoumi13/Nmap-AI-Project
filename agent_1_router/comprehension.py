# Fichier: agent_1_router/comprehension.py
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class ComprehensionAgent:
    def __init__(self, corpus_filename='rag_corpus_detailed.json'):
        print("🧠 Initialisation Comprehension Agent...")
        
        # 1. Chemins Absolus
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.corpus_path = os.path.join(project_root, 'datasets', corpus_filename)
        
        print(f"   📂 Chemin corpus: {self.corpus_path}")

        self.load_rag_corpus()
        
        # 2. Noise Corpus (Bruit)
        self.noise_corpus = [
            "what is the weather like today", "how to bake a chocolate cake",
            "who won the world cup", "play some music", "define relativity theory",
            "how to center a div in css", "install python on windows",
            "fix wifi connection lagging", "what is ram memory",
            "java null pointer exception", "react js tutorial for beginners",
            "git commit push origin master", "configure docker container",
            "reset sql database password", "difference between tcp and udp model",
            "create a website with html", "learn machine learning basics"
        ]
        
        # 3. TF-IDF avec Stop Words
        self.all_corpus = self.nmap_corpus + self.noise_corpus
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            ngram_range=(1, 2), 
            min_df=1,
            stop_words='english' 
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_corpus)
        self.n_nmap = len(self.nmap_corpus)
        
        # 4. Embeddings Sémantiques (CORRIGÉ POUR INCLURE AUTH/EXPLOIT)
        print("   Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.reference_embedding = self.embedding_model.encode(
            "perform network scanning port enumeration service detection vulnerability assessment firewall evasion nse scripting host discovery brute force authentication exploitation password cracking", 
            convert_to_tensor=True
        )
        
        print("   ✅ Comprehension Agent prêt!")

    def load_rag_corpus(self):
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                rag_data = json.load(f)
            
            kb_key = 'knowledge_base' if 'knowledge_base' in rag_data else 'documents'
            documents = rag_data.get(kb_key, [])
            examples = rag_data.get('query_examples', [])

            print(f"   📚 Chargement de {len(documents)} docs et {len(examples)} exemples...")
            
            self.nmap_corpus = []
            self.corpus_metadata = []
            
            for doc in documents:
                text_parts = [
                    doc.get('intent', ''),
                    doc.get('context', ''),
                    doc.get('explanation', ''),
                    ' '.join(doc.get('tags', [])),
                    ' '.join(doc.get('use_cases', [])),       
                    ' '.join(doc.get('related_concepts', [])) 
                ]
                corpus_text = " ".join(text_parts)
                
                self.nmap_corpus.append(corpus_text)
                self.corpus_metadata.append({
                    'id': doc.get('id', 'unknown'),
                    'command': doc.get('command', ''),
                    'category': doc.get('category', 'general'),
                    'difficulty': doc.get('difficulty', 'medium')
                })

            # Chargement des exemples utilisateurs
            for ex in examples:
                user_q = ex.get('user_query', '')
                if user_q:
                    self.nmap_corpus.append(user_q)
                    self.corpus_metadata.append({
                        'id': f"example_for_{ex.get('retrieved_id', '?')}",
                        'command': "See related ID",
                        'category': "example",
                        'difficulty': "n/a"
                    })
            
        except Exception as e:
            print(f"   ❌ Erreur lecture JSON: {e}")
            self._load_default_corpus()

    def _load_default_corpus(self):
        self.nmap_corpus = ["scan network ports tcp udp"]
        self.corpus_metadata = []

    def analyze(self, user_query: str) -> dict:
        # 1. TF-IDF
        query_vec = self.tfidf_vectorizer.transform([user_query])
        all_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        scores_nmap = all_scores[:self.n_nmap]
        scores_noise = all_scores[self.n_nmap:]
        
        max_nmap = np.max(scores_nmap) if len(scores_nmap) > 0 else 0
        max_noise = np.max(scores_noise) if len(scores_noise) > 0 else 0
        
        best_match_idx = int(np.argmax(scores_nmap)) if len(scores_nmap) > 0 else 0
        best_match_metadata = self.corpus_metadata[best_match_idx] if self.corpus_metadata else {}
        
        # 2. Sémantique
        try:
            query_embedding = self.embedding_model.encode(user_query, convert_to_tensor=True)
            from sentence_transformers import util
            semantic_score = float(util.cos_sim(query_embedding, self.reference_embedding)[0][0])
        except:
            semantic_score = max_nmap

        # 3. Score de Base
        final_score = (max_nmap * 0.5) + (semantic_score * 0.5)

        critical_keywords = [
            "scan", "nmap", "port", "ip", "tcp", "udp", "host", "target",
            "os", "fingerprint", "version", "detection",
            "script", "nse", "vuln", "exploit", "brute", "auth",
            "firewall", "ids", "evade", "decoy", "spoof", "mtu", "fragment",
            "stealth", "syn", "ack", "connect", "timing"
        ]
  
        query_lower = user_query.lower()
        if any(word in query_lower for word in critical_keywords):
            final_score += 0.20 
            print("   [BOOST] Mot-clé technique détecté (+0.20)")

        
        if max_noise > max_nmap and max_noise > 0.4:
            final_score -= 0.1
        
      
        THRESHOLD = 0.18 
        
        debug_info = f"TF-IDF: {max_nmap:.2f} | Noise: {max_noise:.2f} | Semantic: {semantic_score:.2f} | Final: {final_score:.2f}"
        print(f"   [DEBUG] {debug_info}")
        
        return {
            "relevant": final_score >= THRESHOLD,
            "score": final_score,
            "reason": "Query matches Nmap context" if final_score >= THRESHOLD else "Irrelevant/Noise",
            "best_match": best_match_metadata if final_score >= THRESHOLD else None
        }
    