import os
import time
import uuid
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
        self.db_path = os.path.join("data", "chroma_db")
        os.makedirs(self.db_path, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection_name = "production_pdf_collection"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        
        # In-memory registry for BM25 indices per document:
        # { doc_id: { "corpus": [chunks], "index": BM25Okapi, "filename": str } }
        self.bm25_indices = {}
        self._rebuild_bm25_from_chroma()

    def _rebuild_bm25_from_chroma(self):
        """Rebuilds in-memory BM25 indices from persistent ChromaDB on startup."""
        if self.collection.count() == 0:
            return

        all_docs = self.collection.get(include=["documents", "metadatas"])
        docs_by_id = {}
        
        for doc_text, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            doc_id = meta.get("doc_id", "default")
            filename = meta.get("source", "unknown.pdf")
            
            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = {"corpus": [], "filename": filename}
            docs_by_id[doc_id]["corpus"].append(doc_text)

        for doc_id, data in docs_by_id.items():
            tokenized_corpus = [text.lower().split() for text in data["corpus"]]
            self.bm25_indices[doc_id] = {
                "corpus": data["corpus"],
                "index": BM25Okapi(tokenized_corpus),
                "filename": data["filename"]
            }

    def ingest_pdf(self, file_path: str, filename: str) -> dict:
        """
        Ingests a PDF without wiping existing knowledge base data.
        Tags every chunk with metadata (doc_id, filename, page_number).
        """
        doc_id = str(uuid.uuid4())[:8]
        doc = fitz.open(file_path)
        
        documents = []
        metadatas = []
        ids = []
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunk_counter = 0

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    documents.append(chunk)
                    metadatas.append({
                        "doc_id": doc_id,
                        "source": filename,
                        "page": page_num + 1
                    })
                    ids.append(f"{doc_id}_chunk_{chunk_counter}")
                    chunk_counter += 1

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            tokenized_corpus = [chunk.lower().split() for chunk in documents]
            self.bm25_indices[doc_id] = {
                "corpus": documents,
                "index": BM25Okapi(tokenized_corpus),
                "filename": filename
            }

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_indexed": len(documents)
        }

    def list_indexed_documents(self) -> list:
        """Returns metadata list of all documents currently indexed."""
        documents_map = {}
        if self.collection.count() > 0:
            all_meta = self.collection.get(include=["metadatas"])["metadatas"]
            for meta in all_meta:
                doc_id = meta.get("doc_id")
                if doc_id and doc_id not in documents_map:
                    documents_map[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get("source", "Unknown")
                    }
        return list(documents_map.values())

    def reciprocal_rank_fusion(self, dense_results: list, sparse_results: list, k: int = 60, top_n: int = 10):
        """Combines Dense Vector and Sparse BM25 ranks via Reciprocal Rank Fusion."""
        doc_scores = {}
        for rank, doc in enumerate(dense_results):
            doc_scores[doc] = doc_scores.get(doc, 0.0) + (1.0 / (k + rank + 1))
        for rank, doc in enumerate(sparse_results):
            doc_scores[doc] = doc_scores.get(doc, 0.0) + (1.0 / (k + rank + 1))

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_n]]

    def rerank_chunks(self, query: str, candidate_chunks: list, top_n: int = 3):
        """Re-ranks candidate context chunks using FlashRank Cross-Encoder."""
        if not candidate_chunks:
            return []
        passages = [{"id": idx, "text": chunk} for idx, chunk in enumerate(candidate_chunks)]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)
        return [result["text"] for result in reranked_results[:top_n]]

    def answer(self, query: str, doc_id: str = None, top_k: int = 3) -> dict:
        """
        Executes multi-document hybrid retrieval, reranking, and generation.
        Supports document-level scoping via doc_id filter.
        """
        start_total = time.time()
        
        if self.collection.count() == 0:
            return {
                "query": query,
                "answer": "No documents are currently indexed in the knowledge base.",
                "retrieved_contexts": [],
                "trace": {}
            }

        # --- Stage 1: Hybrid Search (Dense + Sparse) ---
        t0 = time.time()
        
        where_filter = {"doc_id": doc_id} if (doc_id and doc_id != "all") else None
        
        # Dense Query
        dense_response = self.collection.query(
            query_texts=[query],
            n_results=min(10, self.collection.count()),
            where=where_filter
        )
        dense_chunks = dense_response["documents"][0] if dense_response["documents"] else []

        # Sparse Query
        sparse_chunks = []
        tokenized_query = query.lower().split()
        
        if doc_id and doc_id != "all" and doc_id in self.bm25_indices:
            bm25_info = self.bm25_indices[doc_id]
            sparse_chunks = bm25_info["index"].get_top_n(tokenized_query, bm25_info["corpus"], n=10)
        else:
            for info in self.bm25_indices.values():
                res = info["index"].get_top_n(tokenized_query, info["corpus"], n=5)
                sparse_chunks.extend(res)

        candidate_chunks = self.reciprocal_rank_fusion(dense_chunks, sparse_chunks, top_n=10)
        retrieval_latency = round((time.time() - t0) * 1000, 2)

        # --- Stage 2: FlashRank Cross-Encoder Reranking ---
        t1 = time.time()
        retrieved_contexts = self.rerank_chunks(query, candidate_chunks, top_n=top_k)
        rerank_latency = round((time.time() - t1) * 1000, 2)

        if not retrieved_contexts:
            return {
                "query": query,
                "answer": "I cannot answer this based on the available documentation.",
                "retrieved_contexts": [],
                "trace": {
                    "retrieval_latency_ms": retrieval_latency,
                    "rerank_latency_ms": rerank_latency,
                    "llm_latency_ms": 0.0,
                    "total_latency_ms": round((time.time() - start_total) * 1000, 2)
                }
            }

        # --- Stage 3: LLM Generation ---
        t2 = time.time()
        context_str = "\n\n".join([f"Chunk {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts)])
        
        system_prompt = (
            "You are an enterprise AI document assistant. Answer the user's question using ONLY the provided context chunks. "
            "If the question asks for a general summary or synthesis, combine the details provided across the context chunks. "
            "If the required information is completely missing from the context, reply strictly: "
            "'I cannot answer this based on the available documentation.'"
        )

        completion = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexts:\n{context_str}\n\nQuestion: {query}"}
            ],
            temperature=0.0
        )

        llm_latency = round((time.time() - t2) * 1000, 2)
        total_latency = round((time.time() - start_total) * 1000, 2)

        return {
            "query": query,
            "answer": completion.choices[0].message.content,
            "retrieved_contexts": retrieved_contexts,
            "trace": {
                "num_candidates_retrieved": len(candidate_chunks),
                "num_chunks_reranked": len(retrieved_contexts),
                "retrieval_latency_ms": retrieval_latency,
                "rerank_latency_ms": rerank_latency,
                "llm_latency_ms": llm_latency,
                "total_latency_ms": total_latency
            }
        }