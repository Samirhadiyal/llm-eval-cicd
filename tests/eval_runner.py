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
STATIC_QUALITY_FLOOR = 0.80
MAX_ALLOWED_DEGRADATION = 0.05  # Fail if performance drops > 5% below baseline

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
    print("=" * 65)
    print(f"🚀 STARTING PDF RAG QUALITY GATE & REGRESSION CHECK (Mode: {mode.upper()})")
    print("=" * 65)

    # 1. Load Baseline Metrics for Regression Gate
    baseline_file = os.path.join("data", "baseline_metrics.json")
    baseline_faithfulness = STATIC_QUALITY_FLOOR
    baseline_relevancy = STATIC_QUALITY_FLOOR

    if os.path.exists(baseline_file):
        with open(baseline_file, "r", encoding="utf-8") as f:
            b_data = json.load(f)
            baseline_faithfulness = b_data.get("faithfulness", STATIC_QUALITY_FLOOR)
            baseline_relevancy = b_data.get("relevancy", STATIC_QUALITY_FLOOR)
            print(f"📌 Loaded Production Baseline -> Faithfulness: {baseline_faithfulness:.2f} | Relevancy: {baseline_relevancy:.2f}")

    target_faithfulness = max(STATIC_QUALITY_FLOOR, baseline_faithfulness - MAX_ALLOWED_DEGRADATION)
    target_relevancy = max(STATIC_QUALITY_FLOOR, baseline_relevancy - MAX_ALLOWED_DEGRADATION)

    # 2. Initialize Pipeline & Ingest Sample PDF
    rag = RAGPipeline()
    sample_pdf = os.path.join("data", "sample_eval_doc.pdf")

    if not os.path.exists(sample_pdf):
        print(f"❌ Error: Benchmark PDF missing at '{sample_pdf}'.")
        sys.exit(1)

    print(f"📄 Indexing benchmark PDF: {sample_pdf}...")
    ingest_info = rag.ingest_pdf(sample_pdf, "sample_eval_doc.pdf")
    print(f"✓ Indexed {ingest_info['chunks_indexed']} chunks under doc_id '{ingest_info['doc_id']}'.\n")

    # 3. Load Evaluation Dataset
    eval_dataset_path = os.path.join("data", "eval_dataset.json")
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    if mode == "smoke" or os.getenv("CI") == "true":
        print("⚡ CI/Smoke Mode Active: Running 10 representative test cases...")
        categories = ["Factual Retrieval", "Multi-Chunk Synthesis", "Out-of-Domain Guardrail", "False Premise"]
        selected_cases = []
        for cat in categories:
            cat_cases = [c for c in eval_cases if c.get("category") == cat]
            selected_cases.extend(cat_cases[:3])
        eval_cases = selected_cases[:10]

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    faithfulness_scores = []
    relevancy_scores = []

    print(f"🧪 Evaluating {len(eval_cases)} cases via LLM Judge ({JUDGE_MODEL})...\n")

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

    # Check against both floor and regression tolerance
    passed_floor = avg_faithfulness >= STATIC_QUALITY_FLOOR and avg_relevancy >= STATIC_QUALITY_FLOOR
    passed_regression = avg_faithfulness >= target_faithfulness and avg_relevancy >= target_relevancy
    gate_passed = passed_floor and passed_regression

    print("=" * 65)
    print("📊 FINAL REGRESSION & QUALITY GATE SUMMARY")
    print("=" * 65)
    print(f"Evaluated Test Cases      : {len(eval_cases)}")
    print(f"Current Faithfulness Score: {avg_faithfulness:.2f} (Required Min: {target_faithfulness:.2f})")
    print(f"Current Relevancy Score   : {avg_relevancy:.2f} (Required Min: {target_relevancy:.2f})")
    print(f"Absolute Quality Floor    : {STATIC_QUALITY_FLOOR:.2f}")

    report = {
        "avg_faithfulness": avg_faithfulness,
        "avg_relevancy": avg_relevancy,
        "baseline_faithfulness": baseline_faithfulness,
        "baseline_relevancy": baseline_relevancy,
        "allowed_degradation_tolerance": MAX_ALLOWED_DEGRADATION,
        "total_cases_evaluated": len(eval_cases),
        "mode": mode,
        "status": "PASSED" if gate_passed else "FAILED"
    }
    
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if gate_passed:
        print("\n✅ QUALITY GATE & REGRESSION CHECK PASSED!")
        sys.exit(0)
    else:
        print("\n❌ QUALITY GATE FAILED: Performance regression detected or score below floor.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Quality Gate Evaluation Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 10-case fast CI smoke test")
    parser.add_argument("--full", action="store_true", help="Run full 35-case golden benchmark")
    args = parser.parse_args()

    eval_mode = "smoke" if args.smoke else "full"
    run_evaluation(mode=eval_mode)