import os
import shutil
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "templates", "index.html")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGPipeline()
    yield

app = FastAPI(
    title="Multi-Document Enterprise PDF RAG Service",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = "all"
    top_k: int = 3

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if not os.path.exists(TEMPLATE_PATH):
        raise HTTPException(status_code=404, detail="Template index.html not found.")
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/indexed-documents")
def get_documents():
    """Returns list of all documents currently indexed in ChromaDB."""
    if rag is None:
        return []
    return rag.list_indexed_documents()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Ingests new PDF into knowledge base and cleans up local temp disk copy."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        ingest_result = rag.ingest_pdf(file_path, file.filename)
        
        # Cleanup temporary uploaded disk file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return {
            "status": "success",
            "doc_id": ingest_result["doc_id"],
            "filename": ingest_result["filename"],
            "chunks_indexed": ingest_result["chunks_indexed"],
            "message": f"Successfully indexed {file.filename} into knowledge base ({ingest_result['chunks_indexed']} chunks)!"
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/ask")
def ask_question(request: QueryRequest):
    """Executes multi-document hybrid search, cross-encoder reranking, and generation."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized.")
        
    try:
        return rag.answer(query=request.query, doc_id=request.doc_id, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")