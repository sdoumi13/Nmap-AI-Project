import os
import json
from typing import Dict, Any
# Plus besoin de dotenv pour la clé API car c'est local !

from langchain_chroma import Chroma
# On remplace OpenAI par des outils locaux
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

class NmapRagAgent:
    def __init__(self, dataset_path: str = "nmap_dataset.json", vector_db_path: str = "./chroma_db_local"):
        
        print("[*] Initialisation du mode LOCAL (Ollama + HuggingFace)...")
        self.dataset_path = dataset_path
        self.vector_db_path = vector_db_path
        
        # 1. Embeddings Gratuits (tournent sur ton CPU)
        # "all-MiniLM-L6-v2" est très rapide et léger
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. LLM Local (Ollama)
        # Assure-toi d'avoir fait 'ollama run llama3' avant
        self.llm = ChatOllama(model="llama3", temperature=0)
        
        self.vectorstore = self._initialize_vector_db()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.chain = self._build_chain()

    def _initialize_vector_db(self) -> Chroma:
        # IMPORTANT : Si on change d'embeddings, il faut supprimer l'ancienne DB
        # Si tu as un dossier chroma_db créé avec OpenAI, supprime-le manuellement avant de lancer ce code
        
        if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
            print(f"[*] Chargement de la base vectorielle existante depuis {self.vector_db_path}")
            return Chroma(persist_directory=self.vector_db_path, embedding_function=self.embedding_function)
        
        print(f"[*] Création de la base vectorielle locale à partir de {self.dataset_path}...")
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Le fichier {self.dataset_path} est introuvable.")

        documents = []
        for entry in data:
            doc = Document(
                page_content=entry["input"],
                metadata={"command": entry["output"]}
            )
            documents.append(doc)

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.vector_db_path
        )
        print("[*] Base vectorielle créée avec succès.")
        return vectorstore

    def _build_chain(self):
        # Le prompt reste le même, Llama3 le comprend très bien
        def retrieve_examples(query):
            docs = self.retriever.invoke(query)
            return "\n".join([f"User: {d.page_content}\nAssistant: {d.metadata['command']}" for d in docs])

        system_prompt = """Tu es un expert NMAP. Génère UNIQUEMENT la commande brute demandée.
        Si aucune IP n'est fournie, utilise <TARGET>. Ne donne aucune explication."""

        prompt = ChatPromptTemplate.from_template(
            """{system_prompt}
            
            EXEMPLES DE RÉFÉRENCE :
            {context}
            
            DEMANDE UTILISATEUR : {question}
            COMMANDE NMAP :"""
        )

        chain = (
            {"context": lambda x: retrieve_examples(x["question"]), "question": RunnablePassthrough(), "system_prompt": lambda x: system_prompt}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_query = input_data.get("user_query", "")
        if not user_query: return {"error": "Query vide"}

        try:
            print(f"[RAG Agent Local] Traitement : {user_query}")
            generated_command = self.chain.invoke({"question": user_query})
            
            # Nettoyage fréquent avec les modèles locaux (ils sont parfois bavards)
            generated_command = generated_command.strip().split('\n')[0] 
            
            target_ip = input_data.get("extracted_ip")
            if target_ip and "<TARGET>" in generated_command:
                generated_command = generated_command.replace("<TARGET>", target_ip)
            
            return {
                "nmap_candidate": generated_command,
                "source_agent": "RAG_LOCAL_LLAMA3",
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}