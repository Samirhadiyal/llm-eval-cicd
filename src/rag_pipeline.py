import os
import time
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv

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
        
        # Sparse Index Attributes
        self.bm25_corpus = []
        self.bm25_index = None

        # FlashRank CPU Reranker Model
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def get_or_create_collection(self):
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def process_and_reset_pdf(self, file_path: str, filename: str) -> int:
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        
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

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            self.bm25_corpus = documents
            tokenized_corpus = [doc.lower().split() for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self.bm25_corpus = []
            self.bm25_index = None
        
        self.active_filename = filename
        return len(documents)

    def reciprocal_rank_fusion(self, dense_results: list, sparse_results: list, k: int = 60, top_n: int = 10):
        doc_scores = {}

        for rank, doc in enumerate(dense_results):
            if doc not in doc_scores:
                doc_scores[doc] = 0.0
            doc_scores[doc] += 1.0 / (k + rank + 1)

        for rank, doc in enumerate(sparse_results):
            if doc not in doc_scores:
                doc_scores[doc] = 0.0
            doc_scores[doc] += 1.0 / (k + rank + 1)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_n]]

    def rerank_chunks(self, query: str, candidate_chunks: list, top_n: int = 3):
        if not candidate_chunks:
            return []

        passages = [{"id": idx, "text": chunk} for idx, chunk in enumerate(candidate_chunks)]
        rerank_request = RerankRequest(query=query, passages=passages)
        
        reranked_results = self.reranker.rerank(rerank_request)
        return [result["text"] for result in reranked_results[:top_n]]

    def answer(self, query: str, top_k: int = 3):
        start_total = time.time()
        
        if self.collection.count() == 0:
            return {
                "query": query,
                "answer": "No active PDF loaded. Please upload a PDF document first.",
                "retrieved_contexts": [],
                "active_document": None,
                "trace": {}
            }

        # --- 1. Hybrid Retrieval Stage ---
        t0 = time.time()
        dense_response = self.collection.query(query_texts=[query], n_results=min(10, self.collection.count()))
        dense_chunks = dense_response["documents"][0] if dense_response["documents"] else []

        sparse_chunks = []
        if self.bm25_index:
            tokenized_query = query.lower().split()
            sparse_chunks = self.bm25_index.get_top_n(tokenized_query, self.bm25_corpus, n=min(10, len(self.bm25_corpus)))

        candidate_chunks = self.reciprocal_rank_fusion(dense_chunks, sparse_chunks, top_n=10)
        retrieval_latency = round((time.time() - t0) * 1000, 2)  # in ms

        # --- 2. Reranking Stage ---
        t1 = time.time()
        retrieved_contexts = self.rerank_chunks(query, candidate_chunks, top_n=top_k)
        rerank_latency = round((time.time() - t1) * 1000, 2)  # in ms

        if not retrieved_contexts:
            return {
                "query": query,
                "answer": "I cannot answer this based on the uploaded document.",
                "retrieved_contexts": [],
                "active_document": self.active_filename,
                "trace": {
                    "retrieval_latency_ms": retrieval_latency,
                    "rerank_latency_ms": rerank_latency,
                    "llm_latency_ms": 0,
                    "total_latency_ms": round((time.time() - start_total) * 1000, 2)
                }
            }

        # --- 3. LLM Generation Stage ---
        t2 = time.time()
        context_str = "\n\n".join([f"Chunk {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts)])
        system_prompt = (
            "You are a strict QA assistant. Answer the user's question using ONLY the provided context below. "
            "If the answer cannot be found in the context, reply strictly: 'I cannot answer this based on the available documentation.'"
        )

        user_prompt = f"Contexts:\n{context_str}\n\nQuestion: {query}"

        completion = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        llm_latency = round((time.time() - t2) * 1000, 2)  # in ms
        answer_text = completion.choices[0].message.content
        total_latency = round((time.time() - start_total) * 1000, 2)

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_contexts": retrieved_contexts,
            "active_document": self.active_filename,
            "trace": {
                "num_candidates_retrieved": len(candidate_chunks),
                "num_chunks_reranked": len(retrieved_contexts),
                "retrieval_latency_ms": retrieval_latency,
                "rerank_latency_ms": rerank_latency,
                "llm_latency_ms": llm_latency,
                "total_latency_ms": total_latency
            }
        }