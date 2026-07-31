import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# MUST MATCH INGEST.PY
mlops_docs = "mlops_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class RAGPipeline:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=DB_DIR)
        
        # Use get_or_create_collection to safely handle collection initialization
        self.collection = self.chroma_client.get_or_create_collection(name=mlops_docs)
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.groq_client = Groq(api_key=api_key)

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        contexts = results["documents"][0] if results["documents"] else []
        return contexts

    def answer(self, query: str, top_k: int = 3):
        contexts = self.retrieve(query, top_k=top_k)
        context_str = "\n\n---\n\n".join(contexts) if contexts else "No context available."

        system_prompt = (
            "You are a precise technical assistant. Answer the user question based ONLY on the "
            "provided context below. If the context does not contain enough information to answer, "
            "explicitly state 'I cannot answer this based on the available documentation.' "
            "Do not hallucinate or use external knowledge."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        answer_text = response.choices[0].message.content

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_contexts": contexts
        }