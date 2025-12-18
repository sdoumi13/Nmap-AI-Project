# Fichier: agent_1_router/complexity.py
import json
import os

class ComplexityAgent:
    def __init__(self, 
                 finetuning_filename='finetuning_corpus_detailed.json',
                 diffusion_filename='diffusion_corpus_detailed.json'):
        
        print("🔍 Initialisation Complexity Agent...")
        
        # --- 1. GESTION DES CHEMINS ABSOLUS ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        self.finetuning_path = os.path.join(project_root, 'datasets', finetuning_filename)
        self.diffusion_path = os.path.join(project_root, 'datasets', diffusion_filename)
        
        print(f"   📂 Corpus Fine-tuning: {self.finetuning_path}")
        print(f"   📂 Corpus Diffusion:   {self.diffusion_path}")
        self.hard_keywords = ["script", "nse", "vuln", "exploit", "evade", "bypass", "ipv6", "fragment", "decoy", "spoof"]
        self.medium_keywords = ["os", "version", "service", "udp", "syn", "stealth", "aggressive", "fingerprint", "timing"]
        self.easy_keywords = ["scan", "port", "ping", "check", "host", "find", "discovery"]

        self.load_finetuning_patterns()
        self.load_diffusion_patterns()
        
        print(f"   ✅ Complexity Agent prêt (Hard: {len(self.hard_keywords)}, Medium: {len(self.medium_keywords)})")

    def load_finetuning_patterns(self):
        """Charge les patterns du corpus Medium (Fine-tuning)"""
        try:
            with open(self.finetuning_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            count = 0
            for conv in data.get('conversations', []):
                if conv.get('difficulty') == 'hard':
                    messages = conv.get('turns', conv.get('messages', []))
                    for msg in messages:
                        if msg['role'] == 'user':
                            self._extract_keywords(msg['content'], self.hard_keywords)
                            count += 1
            print(f"   📚 Appris de {count} exemples Fine-tuning.")
            
        except FileNotFoundError:
            print(f"   ⚠️ Fichier Fine-tuning non trouvé.")
        except Exception as e:
            print(f"   ❌ Erreur Fine-tuning JSON: {e}")

    def load_diffusion_patterns(self):
        """Charge les patterns du corpus Hard (Diffusion)"""
        try:
            with open(self.diffusion_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            count = 0
            for item in data.get('training_data', []):
                complexity = item.get('complexity_level', 1)
                if complexity > 6:
                    for tag in item.get('semantic_tags', []):
                        if tag.lower() not in self.hard_keywords:
                            self.hard_keywords.append(tag.lower())
                            count += 1
                    
                    self._extract_keywords(item.get('text_description', ''), self.hard_keywords)

            print(f"   📚 Appris de {count} tags complexes du corpus Diffusion.")
            
        except FileNotFoundError:
            print(f"   ⚠️ Fichier Diffusion non trouvé.")
        except Exception as e:
            print(f"   ❌ Erreur Diffusion JSON: {e}")

    def _extract_keywords(self, text, target_list):
        """Helper pour ajouter des mots intéressants s'ils sont nouveaux"""
        triggers = ["ipv6", "firewall", "ids", "auth", "brute", "mtu", "data-length"]
        for t in triggers:
            if t in text.lower() and t not in target_list:
                target_list.append(t)

    def classify(self, query: str) -> dict:
        q = query.lower()
        
        hard_matches = sum(1 for w in self.hard_keywords if w in q)
        medium_matches = sum(1 for w in self.medium_keywords if w in q)
        
        # 1. HARD (Priorité absolue)
        if hard_matches > 0:
            return {
                "level": "Hard",
                "target_agent": "Agent Hard (Diffusion/ReAct)",
                "reason": f"Complex patterns detected ({hard_matches} matches like '{self._get_match(q, self.hard_keywords)}')",
                "matched_keywords": [w for w in self.hard_keywords if w in q],
                "confidence": min(0.7 + (hard_matches * 0.1), 0.99)
            }
        
        # 2. MEDIUM
        if medium_matches >= 1:
            return {
                "level": "Medium",
                "target_agent": "Agent Medium (Fine-Tuned)",
                "reason": f"Specific options detected ({medium_matches} matches like '{self._get_match(q, self.medium_keywords)}')",
                "matched_keywords": [w for w in self.medium_keywords if w in q],
                "confidence": 0.85
            }
        
        # 3. EASY
        return {
            "level": "Easy",
            "target_agent": "Agent Easy (KG-RAG)",
            "reason": "Simple intent detected",
            "matched_keywords": [],
            "confidence": 0.8
        }

    def _get_match(self, query, keywords):
        """Retourne le premier mot clé trouvé pour l'affichage"""
        for w in keywords:
            if w in query: return w
        return "?"