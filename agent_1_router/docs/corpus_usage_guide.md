# Guide d'Utilisation des Corpus pour les 3 Modèles

## Vue d'Ensemble

Ce document explique comment utiliser les trois corpus détaillés pour entraîner vos modèles RAG, Fine-tuning et Diffusion.

---

## 1. Corpus RAG (Retrieval Augmented Generation)

### 📁 Fichier: `rag_corpus_detailed.json`

### Objectif
Fournir un contexte riche pour la récupération sémantique lors de requêtes utilisateur.

### Structure

```json
{
  "id": "rag_001",
  "category": "basic|discovery|security|stealth|advanced|scripting|ipv6|timing|evasion",
  "difficulty": "easy|medium|hard",
  "intent": "Description courte de l'intention",
  "command": "Commande Nmap exacte",
  "context": "Contexte d'utilisation détaillé",
  "explanation": "Explication technique approfondie",
  "use_cases": ["Cas 1", "Cas 2"],
  "related_concepts": ["Concept 1", "Concept 2"],
  "alternatives": ["Commande alternative 1", "Commande alternative 2"],
  "prerequisites": "Prérequis nécessaires",
  "execution_time": "Durée estimée",
  "tags": ["tag1", "tag2", "tag3"]
}
```

### Utilisation avec RAG

#### Étape 1: Embedding et Indexation
```python
from sentence_transformers import SentenceTransformer
import faiss
import json

# Charger le corpus
with open('rag_corpus_detailed.json', 'r') as f:
    corpus = json.load(f)

# Créer les embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Combiner les champs pour embedding riche
texts_to_embed = []
for item in corpus['knowledge_base']:
    combined_text = f"{item['intent']} {item['context']} {item['explanation']} {' '.join(item['tags'])}"
    texts_to_embed.append(combined_text)

embeddings = model.encode(texts_to_embed)

# Créer index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

#### Étape 2: Récupération lors de Query
```python
def retrieve_relevant_commands(query, k=3):
    # Embed la query
    query_embedding = model.encode([query])
    
    # Recherche dans FAISS
    distances, indices = index.search(query_embedding, k)
    
    # Récupérer les commandes pertinentes
    results = []
    for idx in indices[0]:
        results.append(corpus['knowledge_base'][idx])
    
    return results

# Exemple
query = "scan furtif ipv6 avec scripts"
relevant = retrieve_relevant_commands(query)
```

#### Étape 3: Génération avec Contexte
```python
from openai import OpenAI

def generate_with_rag(user_query):
    # Récupérer contexte
    context_items = retrieve_relevant_commands(user_query, k=3)
    
    # Construire le prompt
    context_str = "\n\n".join([
        f"Commande: {item['command']}\n"
        f"Contexte: {item['context']}\n"
        f"Explication: {item['explanation']}"
        for item in context_items
    ])
    
    prompt = f"""Contexte pertinent:
{context_str}

Question utilisateur: {user_query}

Réponds en utilisant le contexte ci-dessus."""
    
    # Générer réponse
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

### Avantages du Corpus RAG
- ✅ **Contexte enrichi**: Explanations, alternatives, use cases
- ✅ **Métadonnées**: Catégories, difficulté, temps d'exécution
- ✅ **Recherche multi-critères**: Tags, concepts, intentions
- ✅ **Évolutif**: Facile d'ajouter de nouvelles entrées

---

## 2. Corpus Fine-tuning (Instruction-Response)

### 📁 Fichier: `finetuning_corpus_detailed.json`

### Objectif
Entraîner des LLMs à répondre comme un expert Nmap avec raisonnement pédagogique.

### Structure

```json
{
  "conversation_id": "ft_001",
  "category": "basic|security|stealth|scripting|evasion",
  "difficulty": "easy|medium|hard",
  "turns": [
    {
      "role": "system",
      "content": "Prompt système définissant l'expertise"
    },
    {
      "role": "user",
      "content": "Question utilisateur"
    },
    {
      "role": "assistant",
      "content": "Réponse détaillée avec commandes, explications, avertissements"
    }
  ]
}
```

### Utilisation pour Fine-tuning

#### Format OpenAI Fine-tuning
```python
import json

def convert_to_openai_format(conversations):
    """Convertir en format JSONL pour OpenAI fine-tuning"""
    output = []
    
    for conv in conversations:
        # Chaque conversation devient une ligne
        training_example = {
            "messages": conv["turns"]
        }
        output.append(training_example)
    
    # Sauvegarder en JSONL
    with open('training_data.jsonl', 'w') as f:
        for example in output:
            f.write(json.dumps(example) + '\n')

# Charger et convertir
with open('finetuning_corpus_detailed.json', 'r') as f:
    data = json.load(f)

convert_to_openai_format(data['conversations'])
```

#### Fine-tuning avec OpenAI API
```python
from openai import OpenAI

client = OpenAI()

# Upload training file
with open('training_data.jsonl', 'rb') as f:
    training_file = client.files.create(
        file=f,
        purpose='fine-tune'
    )

# Créer le fine-tune job
fine_tune_job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": 0.5
    }
)

print(f"Fine-tune job created: {fine_tune_job.id}")
```

#### Format pour Llama / Mistral
```python
def convert_to_llama_format(conversations):
    """Format pour Llama avec tokens spéciaux"""
    output = []
    
    for conv in conversations:
        formatted = ""
        for turn in conv["turns"]:
            if turn["role"] == "system":
                formatted += f"<|system|>\n{turn['content']}\n"
            elif turn["role"] == "user":
                formatted += f"<|user|>\n{turn['content']}\n"
            elif turn["role"] == "assistant":
                formatted += f"<|assistant|>\n{turn['content']}\n"
        
        output.append({
            "text": formatted,
            "metadata": {
                "category": conv["category"],
                "difficulty": conv["difficulty"]
            }
        })
    
    return output
```

### Stratégies d'Entraînement

#### Paramètres Recommandés
```python
training_config = {
    "epochs": 3-5,
    "learning_rate": 1e-5,  # Très petit pour éviter oubli catastrophique
    "batch_size": 4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4
}
```

#### Validation et Monitoring
```python
def evaluate_model(model, test_conversations):
    """Évaluer la qualité des réponses générées"""
    scores = []
    
    for conv in test_conversations:
        # Prendre seulement user messages
        user_msg = [turn for turn in conv["turns"] if turn["role"] == "user"][0]
        
        # Générer réponse
        response = model.generate(user_msg["content"])
        
        # Scorer (vous pouvez utiliser BLEU, ROUGE, ou un modèle de scoring)
        score = compute_quality_score(response, conv["turns"][-1]["content"])
        scores.append(score)
    
    return sum(scores) / len(scores)
```

### Avantages du Corpus Fine-tuning
- ✅ **Multi-turn conversations**: Apprend le contexte progressif
- ✅ **Raisonnement expert**: Explications détaillées et pédagogiques
- ✅ **Awareness sécurité**: Avertissements légaux et éthiques intégrés
- ✅ **Format flexible**: Compatible OpenAI, Llama, Mistral

---

## 3. Corpus Diffusion Models (Text-to-Command)

### 📁 Fichier: `diffusion_corpus_detailed.json`

### Objectif
Entraîner des modèles de diffusion à générer des commandes Nmap valides à partir de descriptions en langage naturel.

### Structure

```json
{
  "id": "diff_001",
  "text_description": "Description détaillée en langage naturel de l'intention",
  "text_embedding_context": "Mots-clés et concepts pour embedding",
  "target_command": "nmap -p- 192.168.1.0/24",
  "complexity_level": 1-10,
  "semantic_tags": ["tag1", "tag2"],
  "intent_category": "comprehensive_scanning",
  "command_structure": {
    "tool": "nmap",
    "port_spec": "-p-",
    "target": "192.168.1.0/24",
    "additional_flags": []
  }
}
```

### Architecture Diffusion Recommandée

#### Approche 1: Latent Diffusion pour Texte
```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class NmapCommandDiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder pour description text
        self.text_encoder = T5Tokenizer.from_pretrained('t5-base')
        
        # Decoder pour commandes
        self.command_decoder = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Diffusion components
        self.noise_schedule = self.create_noise_schedule(1000)
    
    def create_noise_schedule(self, timesteps):
        """Créer le schedule de bruit pour diffusion"""
        betas = torch.linspace(0.0001, 0.02, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod
    
    def forward_diffusion(self, x, t):
        """Ajouter bruit progressivement"""
        noise = torch.randn_like(x)
        alpha_t = self.noise_schedule[t]
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
    
    def reverse_diffusion(self, text_condition, num_steps=50):
        """Générer commande depuis bruit"""
        # Encoder la description
        encoded = self.text_encoder(text_condition, return_tensors='pt')
        
        # Démarrer depuis bruit pur
        x = torch.randn(1, 512)  # Latent dimension
        
        # Denoising progressif
        for t in reversed(range(num_steps)):
            # Prédire le bruit
            predicted_noise = self.command_decoder(
                inputs_embeds=x,
                encoder_hidden_states=encoded.last_hidden_state
            )
            
            # Retirer bruit
            alpha_t = self.noise_schedule[t]
            x = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        # Décoder en commande
        command = self.decode_to_command(x)
        return command
```

#### Approche 2: Seq2Seq avec Conditioning
```python
from transformers import BartForConditionalGeneration, BartTokenizer

class CommandGeneratorModel:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    def train(self, training_data):
        """Fine-tuner BART sur corpus diffusion"""
        for item in training_data:
            # Input: description + contexte
            input_text = f"{item['text_description']} [SEP] {item['text_embedding_context']}"
            
            # Output: commande
            output_text = item['target_command']
            
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
            outputs = self.tokenizer(output_text, return_tensors='pt', max_length=128, truncation=True)
            
            # Training step
            loss = self.model(
                input_ids=inputs.input_ids,
                labels=outputs.input_ids
            ).loss
            
            loss.backward()
    
    def generate_command(self, description, num_beams=5):
        """Générer commande depuis description"""
        inputs = self.tokenizer(description, return_tensors='pt')
        
        outputs = self.model.generate(
            inputs.input_ids,
            num_beams=num_beams,
            max_length=128,
            early_stopping=True,
            num_return_sequences=3  # Retourner top 3
        )
        
        commands = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return commands
```

### Entraînement du Modèle Diffusion

```python
import json
from torch.utils.data import Dataset, DataLoader

class NmapDiffusionDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)['training_data']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'description': item['text_description'],
            'context': item['text_embedding_context'],
            'command': item['target_command'],
            'complexity': item['complexity_level']
        }

# Créer dataset
dataset = NmapDiffusionDataset('diffusion_corpus_detailed.json')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
model = CommandGeneratorModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(10):
    for batch in dataloader:
        # Forward pass
        loss = model.compute_loss(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### Validation de Commandes Générées

```python
import re

def validate_nmap_command(command):
    """Valider qu'une commande générée est syntaxiquement correcte"""
    checks = {
        "starts_with_nmap": command.strip().startswith('nmap'),
        "has_target": bool(re.search(r'\d+\.\d+\.\d+\.\d+', command) or re.search(r'[\da-f:]+::', command)),
        "valid_flags": all(flag in ['-p', '-sS', '-sV', '-O', '-A', '-T', '-sn', '--script', '-6', '-f', '-D'] 
                          for flag in re.findall(r'-[a-zA-Z]+', command)),
        "no_syntax_errors": not re.search(r'--\s+--', command)  # Pas de double flags
    }
    
    return all(checks.values()), checks

# Test
generated_command = "nmap -sS -p 80,443 192.168.1.100"
is_valid, details = validate_nmap_command(generated_command)
print(f"Valid: {is_valid}, Details: {details}")
```

### Avantages du Corpus Diffusion
- ✅ **Descriptions riches**: Langage naturel varié
- ✅ **Contexte embedding**: Mots-clés pour améliorer conditioning
- ✅ **Structure commande**: Parsing facilité
- ✅ **Niveaux complexité**: Entraînement progressif

---

## 🔄 Workflow Complet Intégré

### Scénario: Système Multi-Modèles

```python
class NmapAISystem:
    def __init__(self):
        # Charger les 3 corpus
        self.rag_corpus = self.load_corpus('rag_corpus_detailed.json')
        self.finetuned_model = self.load_finetuned_model('nmap_expert_model')
        self.diffusion_model = self.load_diffusion_model('command_generator')
    
    def process_query(self, user_query):
        """Pipeline complet"""
        
        # 1. Routage intelligent (comme dans vos logs)
        difficulty = self.classify_difficulty(user_query)
        
        if difficulty == "easy":
            # Utiliser RAG direct
            return self.rag_pipeline(user_query)
        
        elif difficulty == "medium":
            # Utiliser modèle fine-tuné
            return self.finetuned_pipeline(user_query)
        
        else:  # hard
            # Utiliser diffusion pour génération créative
            return self.diffusion_pipeline(user_query)
    
    def rag_pipeline(self, query):
        """Pipeline RAG pour queries simples"""
        # Récupérer contexte pertinent
        context = retrieve_relevant_commands(query, k=3)
        
        # Générer réponse avec LLM standard + contexte
        response = generate_with_rag(query)
        
        return {
            "command": context[0]['command'],
            "explanation": response,
            "method": "RAG"
        }
    
    def finetuned_pipeline(self, query):
        """Pipeline fine-tuned pour queries moyennes"""
        # Le modèle a appris les patterns
        response = self.finetuned_model.generate(query)
        
        # Extraire commande de la réponse
        command = self.extract_command(response)
        
        return {
            "command": command,
            "explanation": response,
            "method": "Fine-tuned"
        }
    
    def diffusion_pipeline(self, query):
        """Pipeline diffusion pour queries complexes"""
        # Générer plusieurs candidats
        candidates = self.diffusion_model.generate_command(query, num_return_sequences=5)
        
        # Valider et ranker
        valid_commands = []
        for cmd in candidates:
            is_valid, _ = validate_nmap_command(cmd)
            if is_valid:
                valid_commands.append(cmd)
        
        # Retourner le meilleur
        best_command = valid_commands[0] if valid_commands else "Error: No valid command generated"
        
        # Utiliser RAG pour explication
        explanation = self.generate_explanation(best_command)
        
        return {
            "command": best_command,
            "explanation": explanation,
            "method": "Diffusion",
            "alternatives": valid_commands[1:3]
        }
```

---

## 📊 Métriques d'Évaluation

### Pour RAG
```python
def evaluate_rag_system(test_queries):
    metrics = {
        "retrieval_precision": [],
        "response_relevance": [],
        "latency": []
    }
    
    for query in test_queries:
        start_time = time.time()
        
        # Retrieval
        retrieved = retrieve_relevant_commands(query)
        
        # Évaluer précision
        precision = compute_precision(retrieved, query.ground_truth)
        metrics["retrieval_precision"].append(precision)
        
        # Latence
        latency = time.time() - start_time
        metrics["latency"].append(latency)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

### Pour Fine-tuning
```python
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

def evaluate_finetuned_model(model, test_conversations):
    rouge = Rouge()
    bleu_scores = []
    rouge_scores = []
    
    for conv in test_conversations:
        ground_truth = conv["turns"][-1]["content"]
        generated = model.generate(conv["turns"][0]["content"])
        
        # BLEU
        bleu = sentence_bleu([ground_truth.split()], generated.split())
        bleu_scores.append(bleu)
        
        # ROUGE
        scores = rouge.get_scores(generated, ground_truth)[0]
        rouge_scores.append(scores['rouge-l']['f'])
    
    return {
        "bleu": np.mean(bleu_scores),
        "rouge-l": np.mean(rouge_scores)
    }
```

### Pour Diffusion
```python
def evaluate_diffusion_model(model, test_data):
    metrics = {
        "syntax_accuracy": [],
        "semantic_similarity": [],
        "execution_success": []
    }
    
    for item in test_data:
        # Générer commande
        generated = model.generate_command(item['text_description'])
        ground_truth = item['target_command']
        
        # Syntax correctness
        is_valid, _ = validate_nmap_command(generated)
        metrics["syntax_accuracy"].append(int(is_valid))
        
        # Semantic similarity
        similarity = compute_semantic_similarity(generated, ground_truth)
        metrics["semantic_similarity"].append(similarity)
        
        # Test si exécutable (simulation)
        success = test_command_execution(generated)
        metrics["execution_success"].append(int(success))
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

---

