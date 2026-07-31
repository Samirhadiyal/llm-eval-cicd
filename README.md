# Automated LLM Evaluation & Regression CI/CD Pipeline

A production-grade MLOps evaluation pipeline for Retrieval-Augmented Generation (RAG) applications. Every code push or context update automatically triggers a 35-point evaluation suite in GitHub Actions to catch hallucinations and quality regressions before shipping to production.

![CI Status](https://github.com/YOUR_GITHUB_USERNAME/llm-eval-cicd/actions/workflows/eval_ci.yml/badge.svg)

---

## 💡 Problem & Solution

* **The Problem:** RAG pipelines are prone to subtle regressions. Changing prompt templates, retrieval parameters (k-value), or chunking strategies can introduce silent hallucinations without throwing runtime errors.
* **The Solution:** An automated CI/CD pipeline using **LLM-as-a-Judge** scoring. Every pull request is evaluated on **Faithfulness** and **Answer Relevancy**. If quality drops below our defined threshold (`0.80`), the build fails automatically.

---

## 🏗️ Architecture

```text
[ Document Source ] ---> [ Chunk & Embed (all-MiniLM-L6-v2) ] ---> [ ChromaDB Vector Store ]
                                                                             │
[ Query Request ] ───────────────────────────────────────────────────────────┼──► [ Context Retrieval ]
                                                                             │           │
[ FastAPI /ask ] ◄─── [ Groq / Llama 3.1 8B Generation ] ◄───────────────────┘           │
                                                                                         │
─────────────────────────────────────────────────────────────────────────────────────────┼──
                                                                                         ▼
                                                                     [ LLM Judge (Llama 3.3 70B) ]
                                                                                 │
                                                                                 ▼
                                                                  [ CI/CD Quality Gate Threshold ]