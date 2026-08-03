import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection_name = "active_pdf_collection"
        self.collection = self.get_or_create_collection()
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.active_filename = None

    def get_or_create_collection(self):
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def process_and_reset_pdf(self, file_path: str, filename: str) -> int:
        """Option 1: Wipe existing collection and index ONLY the new PDF."""
        # 1. Reset / Delete existing collection to prevent cross-document contamination
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # If it doesn't exist yet
        
        # 2. Re-create clean empty collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        
        # 3. Extract text page-by-page from PDF
        doc = fitz.open(file_path)
        documents = []
        metadatas = []
        ids = []
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunk_id = 0

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    documents.append(chunk)
                    metadatas.append({"page": page_num + 1, "source": filename})
                    ids.append(f"chunk_{chunk_id}")
                    chunk_id += 1

        # 4. Insert chunks into ChromaDB if text was found
        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        self.active_filename = filename
        return len(documents)

    def answer(self, query: str, top_k: int = 3):
        if self.collection.count() == 0:
            return {
                "query": query,
                "answer": "No active PDF loaded. Please upload a PDF document first.",
                "retrieved_contexts": [],
                "active_document": None
            }

        # Retrieve top K matching chunks
        results = self.collection.query(query_texts=[query], n_results=top_k)
        retrieved_contexts = results["documents"][0] if results["documents"] else []

        if not retrieved_contexts:
            return {
                "query": query,
                "answer": "I cannot answer this based on the uploaded document.",
                "retrieved_contexts": [],
                "active_document": self.active_filename
            }

        # Build Strict System Prompt
        context_str = "\n\n".join([f"Chunk {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts)])
        system_prompt = (
            "You are a strict QA assistant. Answer the user's question using ONLY the provided context below. "
            "If the answer cannot be found in the context, reply strictly: 'I cannot answer this based on the available documentation.'"
        )

        user_prompt = f"Contexts:\n{context_str}\n\nQuestion: {query}"

        # Inference via Groq API
        completion = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        answer_text = completion.choices[0].message.content

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_contexts": retrieved_contexts,
            "active_document": self.active_filename
        }