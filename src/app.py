import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

# Load variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "templates", "index.html")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Dynamic PDF RAG Service", version="2.0.0")
rag = None

@app.on_event("startup")
def startup_event():
    global rag
    rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if not os.path.exists(TEMPLATE_PATH):
        raise HTTPException(status_code=404, detail="Template index.html not found.")
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    chunks_count = rag.process_and_reset_pdf(file_path, file.filename)
    
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": chunks_count,
        "message": f"Previous knowledge wiped. Successfully indexed {file.filename}!"
    }

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        return rag.answer(query=request.query, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))