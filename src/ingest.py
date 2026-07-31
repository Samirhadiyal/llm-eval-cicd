import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = os.path.join(BASE_DIR, "data", "docs", "mlops_playbook.md")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "mlops_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def run_ingestion():
    if not os.path.exists(DOC_PATH):
        print(f"Error: {DOC_PATH} does not exist.")
        return

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        doc_text = f.read()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40,
        separators=["\n\n", "\n", "- ", " "]
    )
    chunks = text_splitter.split_text(doc_text)
    print(f"Created {len(chunks)} text chunks.")

    # Embed chunks using HuggingFace local model (free CPU)
    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()

    # Persist in ChromaDB
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Reset collection if exists to allow fresh re-ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    
    ids = [f"chunk_{i:03d}" for i in range(len(chunks))]
    metadatas = [{"source": "mlops_playbook.md"} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Ingestion successful! {len(chunks)} chunks saved to '{DB_DIR}'.")

if __name__ == "__main__":
    run_ingestion()