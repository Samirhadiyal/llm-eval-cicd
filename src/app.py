import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "templates", "index.html")

app = FastAPI(title="LLM RAG Service", version="1.0.0")

rag = None

@app.on_event("startup")
def startup_event():
    global rag
    rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_contexts: list[str]

# Serve HTML Web Interface on Root Route
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if not os.path.exists(TEMPLATE_PATH):
        raise HTTPException(status_code=404, detail="Template index.html not found.")
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        result = rag.answer(query=request.query, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))