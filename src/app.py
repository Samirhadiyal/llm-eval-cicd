import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "templates", "index.html")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global RAG Pipeline instance
rag = None

# Modern FastAPI Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    # Initialize pipeline on startup
    rag = RAGPipeline()
    yield
    # Cleanup on shutdown (if needed)

app = FastAPI(
    title="Dynamic PDF RAG Service with Observability",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the frontend user interface."""
    if not os.path.exists(TEMPLATE_PATH):
        raise HTTPException(status_code=404, detail="Template index.html not found.")
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process into vector store
        chunks_count = rag.process_and_reset_pdf(file_path, file.filename)
        
        # Cleanup uploaded PDF file from server disk after indexing
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": chunks_count,
            "message": f"Previous knowledge wiped. Successfully indexed {file.filename}!"
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Executes 2-Stage Retrieval (Hybrid BM25/Vector Search + FlashRank Reranking)
    and returns answer with full step-by-step latency trace telemetry.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized.")
        
    try:
        # Returns answer, retrieved contexts, active document, and latency trace dictionary
        return rag.answer(query=request.query, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")