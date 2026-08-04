# 🛡️ LLM Eval CI/CD Pipeline

**A multi-document RAG service that treats its own evaluator as a first-class engineering problem** — hybrid retrieval, cross-encoder reranking, and a CI/CD quality gate that checks for regression drift, not just a pass/fail floor.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-ff6600?style=for-the-badge&logo=databricks&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1/3.3-f34f29?style=for-the-badge&logo=speedtest&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

---

## Screenshot
<img width="1536" height="861" alt="HOME" src="https://github.com/user-attachments/assets/5fe96412-f5c0-4811-9d00-227643184c3f" />
<br><br>

<img width="1517" height="864" alt="Query_1" src="https://github.com/user-attachments/assets/1208d813-5bd6-4fe7-a9e6-836eba256b87" />
<br><br>

<img width="1521" height="859" alt="Query_2" src="https://github.com/user-attachments/assets/749ff7c6-9f06-4604-92ba-ce5e6094c2b1" />
<br><br>

<img width="1516" height="864" alt="Wrong_Query " src="https://github.com/user-attachments/assets/06b3b007-4c8f-4ff9-b065-f1c35febfc32" />
<br><br>
---

## The Problem

Traditional software fails loudly — a broken build throws an error and stops the pipeline. RAG applications fail silently. Change a chunk size, swap an embedding model, or tweak a prompt, and the system will keep returning fluent, confident, **wrong** answers with an HTTP 200. Nothing in a standard CI pipeline catches that.

Most RAG portfolio projects stop at "it works when I tested it manually." This project treats that as the actual engineering problem: **if you don't evaluate your evaluator, your quality gate is just vibes with extra steps.**

---

## What's Actually Here

| Layer | What it does |
|---|---|
| **Hybrid Retrieval** | Dense search (ChromaDB + `all-MiniLM-L6-v2`) fused with sparse keyword search (BM25Okapi) via Reciprocal Rank Fusion |
| **Cross-Encoder Reranking** | Top-10 RRF candidates re-scored by a FlashRank (`ms-marco-MiniLM-L-12-v2`) cross-encoder before hitting the LLM |
| **Multi-Document Scope** | Documents are tagged with a `doc_id` at ingest; queries can target one document or search across all indexed documents |
| **LLM-as-Judge Evaluation** | A separate model (`llama-3.3-70b-versatile`) scores every generation on Faithfulness and Relevancy, independent of the generation model (`llama-3.1-8b-instant`) |
| **Judge Calibration** | The judge itself is checked against a hand-labeled ground-truth set before being trusted as a gate — see [caveats](#judge-calibration--honest-scope) below |
| **Regression-Aware CI Gate** | GitHub Actions blocks a merge not just when quality drops below an absolute floor, but when it drifts more than 5% below the last known-good baseline |
| **Tiered Test Execution** | `--smoke` (10 cases, CI-fast) vs `--full` (all 35 cases, pre-release) |

---

## Architecture

```
                    ┌─────────────────────┐
   PDF Upload  ───▶ │  Chunking (500/50)   │
                    └──────────┬───────────┘
                               ▼
              ┌────────────────────────────────┐
              │  ChromaDB (dense)  +  BM25 (sparse) │
              └────────────────┬───────────────┘
                               ▼
                  Reciprocal Rank Fusion (top-10)
                               ▼
                  FlashRank Cross-Encoder Rerank (top-3)
                               ▼
                    Groq · llama-3.1-8b-instant
                               ▼
                         Answer + Trace
```

Every request returns a `trace` object with real measured latency per stage (retrieval, rerank, generation) — not estimated, not hardcoded.

---

## The CI/CD Quality Gate

```yaml
on: [push, pull_request]
run: python tests/eval_runner.py --smoke   # 10 cases, gated on every PR
```

The gate checks two things, not one:

1. **Absolute floor** — Faithfulness and Relevancy must each be ≥ 0.80
2. **Regression tolerance** — score must not drop more than 5% below the stored production baseline (`data/baseline_metrics.json`)

A static floor alone can't catch a real regression (e.g., going from 0.98 to 0.81 still "passes" a 0.80 floor but is a meaningful quality drop). This gate catches both failure modes.

```
📊 FINAL REGRESSION & QUALITY GATE SUMMARY
Evaluated Test Cases      : 10
Current Faithfulness Score: 1.00 (Required Min: 0.95)
Current Relevancy Score   : 1.00 (Required Min: 0.95)
Absolute Quality Floor    : 0.80

✅ QUALITY GATE & REGRESSION CHECK PASSED!
```

The 35-case golden benchmark spans four categories designed to stress different failure modes, not just "does it retrieve the right chunk":

| Category | Cases | Tests |
|---|---|---|
| Direct Factual Retrieval | 10 | Single-fact lookup accuracy |
| Multi-Chunk Synthesis | 10 | Reasoning across multiple retrieved chunks |
| Out-of-Domain Guardrail | 8 | Clean refusal on questions outside the document's scope |
| False Premise | 7 | Correcting or refusing queries built on a false assumption about the document |

---

## Judge Calibration — Honest Scope

Before trusting an LLM to gate deployments, I checked whether it agrees with a human on cases where the correct answer is unambiguous — a hallucination, a correct refusal, a correct fact. `tests/calibrate_judge.py` runs 5 hand-labeled sanity-check cases (clear hallucination, clear refusal, clear factual match) and compares the judge's score against the known-correct label.

**Current result: 100% agreement on this 5-case sanity set.**

I'm stating this precisely rather than just citing the percentage, because a 5-case check with unambiguous cases is a sanity check, not a statistically rigorous calibration — and I'd rather a reviewer hear that from me than discover it themselves. A properly powered calibration would need a larger set (30+ cases) including genuinely borderline judgments, not just clear-cut ones. That's the next thing I'd build on this project, not a gap I'm pretending doesn't exist.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload-pdf` | POST | Ingest a PDF, assign `doc_id`, index chunks (dense + sparse), auto-delete temp file after processing |
| `/ask` | POST | Query with `doc_id` (specific document or `"all"`) and `top_k` |
| `/indexed-documents` | GET | List all currently indexed documents |
| `/` | GET | Web UI with scope selection and per-request latency telemetry |

---

## Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| Vector DB | ChromaDB (persistent) |
| Sparse Search | BM25Okapi (`rank-bm25`), document-scoped |
| Fusion | Reciprocal Rank Fusion |
| Reranker | FlashRank (`ms-marco-MiniLM-L-12-v2`), CPU |
| Generation | Groq — `llama-3.1-8b-instant` |
| Judge | Groq — `llama-3.3-70b-versatile` |
| CI/CD | GitHub Actions |

---

## Run It Locally

```bash
git clone https://github.com/Samirhadiyal/llm-eval-cicd.git
cd llm-eval-cicd
pip install -r requirements.txt

# Add GROQ_API_KEY to a .env file

uvicorn src.app:app --reload
# UI:    http://127.0.0.1:8000/
# Docs:  http://127.0.0.1:8000/docs

# Run the eval suite
python tests/eval_runner.py --smoke   # fast, 10 cases
python tests/eval_runner.py --full    # complete, 35 cases

# Check judge calibration
python tests/calibrate_judge.py
```

---

## What I'd Build Next

- Expand judge calibration from 5 sanity cases to 30+ with genuinely ambiguous/borderline examples
- Concurrent-upload handling for the multi-doc ingest path
- Persistent, non-ephemeral storage for production deployment (current setup assumes disk resets between sessions)
- A second, independent judge model to cross-check the primary judge's scores against each other, not just against a human set
