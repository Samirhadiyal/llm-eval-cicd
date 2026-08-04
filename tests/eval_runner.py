import os
import sys
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import Groq
from src.rag_pipeline import RAGPipeline

JUDGE_MODEL = "llama-3.3-70b-versatile"
FAITHFULNESS_THRESHOLD = 0.80
RELEVANCY_THRESHOLD = 0.80

def evaluate_with_judge(groq_client, query: str, context: str, answer: str, category: str = "General"):
    prompt = f"""
    You are an impartial MLOps evaluation judge scoring an enterprise RAG application.
    
    User Query Category: {category}
    User Query: {query}
    Retrieved Context: {context}
    Generated Answer: {answer}
    
    Evaluate the response on two metrics from 0.0 to 1.0:
    1. Faithfulness: Is the answer strictly derived ONLY from the provided context without making up outside facts? (1.0 = faithful, 0.0 = hallucination).
    2. Relevancy: Does the answer appropriately address the user's query? 
       CRITICAL RULE: If the query is out-of-domain, unanswerable from context, or based on a false premise, and the model outputs the standard refusal message ("I cannot answer this based on the available documentation."), this IS the expected correct response and MUST receive Relevancy = 1.0.

    Return JSON ONLY with this exact schema:
    {{
      "faithfulness": 0.0 to 1.0,
      "relevancy": 0.0 to 1.0,
      "reasoning": "Brief 1-sentence justification"
    }}
    """
    
    try:
        response = groq_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Judge evaluation error: {e}")
        return {"faithfulness": 0.0, "relevancy": 0.0, "reasoning": "Judge execution failed"}

def run_evaluation(mode: str = "full"):
    print("=" * 60)
    print(f"🚀 Starting PDF RAG Quality Gate Evaluation (Mode: {mode.upper()})")
    print("=" * 60)

    # 1. Initialize Pipeline & Ingest Sample PDF
    rag = RAGPipeline()
    sample_pdf = os.path.join("data", "sample_eval_doc.pdf")

    if not os.path.exists(sample_pdf):
        print(f"❌ Error: Benchmark PDF missing at '{sample_pdf}'. Please run generate script.")
        sys.exit(1)

    print(f"📄 Indexing benchmark PDF: {sample_pdf}...")
    ingest_info = rag.ingest_pdf(sample_pdf, "sample_eval_doc.pdf")
    print(f"✓ Indexed {ingest_info['chunks_indexed']} chunks under doc_id '{ingest_info['doc_id']}'.\n")

    # 2. Load Evaluation Dataset
    eval_dataset_path = os.path.join("data", "eval_dataset.json")
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    # If Smoke Mode or CI, select 10 balanced cases across categories
    if mode == "smoke" or os.getenv("CI") == "true":
        print("⚡ Running in CI/Smoke Mode: Selecting 10 representative test cases...")
        categories = ["Factual Retrieval", "Multi-Chunk Synthesis", "Out-of-Domain Guardrail", "False Premise"]
        selected_cases = []
        for cat in categories:
            cat_cases = [c for c in eval_cases if c.get("category") == cat]
            selected_cases.extend(cat_cases[:3])  # Take top 2-3 per category
        eval_cases = selected_cases[:10]

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    faithfulness_scores = []
    relevancy_scores = []

    print(f"🧪 Running {len(eval_cases)} evaluation cases through LLM Judge ({JUDGE_MODEL})...\n")

    for idx, case in enumerate(eval_cases, 1):
        query = case["query"]
        category = case.get("category", "General")

        result = rag.answer(query=query, doc_id=ingest_info["doc_id"])
        answer = result["answer"]
        contexts = "\n".join(result["retrieved_contexts"])

        eval_result = evaluate_with_judge(groq_client, query, contexts, answer, category=category)
        
        f_score = float(eval_result.get("faithfulness", 0.0))
        r_score = float(eval_result.get("relevancy", 0.0))

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)

        print(f"[{idx}/{len(eval_cases)}] Category: {category}")
        print(f"  Q: {query}")
        print(f"  A: {answer[:80]}...")
        print(f"  Score -> Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}")
        print(f"  Reason: {eval_result.get('reasoning')}\n")
        
        time.sleep(0.3)

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)

    print("=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Evaluated Cases         : {len(eval_cases)}")
    print(f"Average Faithfulness Score: {avg_faithfulness:.2f} (Target: >= {FAITHFULNESS_THRESHOLD})")
    print(f"Average Relevancy Score   : {avg_relevancy:.2f} (Target: >= {RELEVANCY_THRESHOLD})")

    report = {
        "avg_faithfulness": avg_faithfulness,
        "avg_relevancy": avg_relevancy,
        "threshold": FAITHFULNESS_THRESHOLD,
        "total_cases_evaluated": len(eval_cases),
        "mode": mode,
        "status": "PASSED" if avg_faithfulness >= FAITHFULNESS_THRESHOLD and avg_relevancy >= RELEVANCY_THRESHOLD else "FAILED"
    }
    
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if avg_faithfulness >= FAITHFULNESS_THRESHOLD and avg_relevancy >= RELEVANCY_THRESHOLD:
        print("\n✅ QUALITY GATE PASSED: PDF RAG Pipeline meets quality thresholds!")
        sys.exit(0)
    else:
        print("\n❌ QUALITY GATE FAILED: Performance fell below target quality thresholds.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Quality Gate Evaluation Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 10-case fast CI smoke test")
    parser.add_argument("--full", action="store_true", help="Run full 35-case golden benchmark")
    args = parser.parse_args()

    eval_mode = "smoke" if args.smoke else "full"
    run_evaluation(mode=eval_mode)