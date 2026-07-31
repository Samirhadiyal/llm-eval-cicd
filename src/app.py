from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline

app = FastAPI(title="LLM RAG Service", version="1.0.0")

# Initialize pipeline lazily or on startup
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

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "RAG API"}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        result = rag.answer(query=request.query, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))