import os
import sys
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure src module is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import Groq
from src.rag_pipeline import RAGPipeline

JUDGE_MODEL = "llama-3.3-70b-versatile"
FAITHFULNESS_THRESHOLD = 0.80
RELEVANCY_THRESHOLD = 0.80

def evaluate_with_judge(groq_client, query: str, context: str, answer: str):
    prompt = f"""
    You are an impartial MLOps evaluation judge scoring a RAG application.
    
    User Query: {query}
    Retrieved Context: {context}
    Generated Answer: {answer}
    
    Evaluate the response on two metrics from 0.0 to 1.0:
    1. Faithfulness: Is the answer strictly derived ONLY from the provided context without making up outside facts?
    2. Relevancy: Does the answer directly address the user's query? If the query is out-of-domain and the model safely refuses, give high relevancy (1.0).

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

def run_evaluation():
    print("=" * 60)
    print("🚀 Starting Automated PDF RAG CI/CD Quality Gate")
    print("=" * 60)

    # 1. Initialize Pipeline & Ingest Sample PDF
    rag = RAGPipeline()
    sample_pdf = os.path.join("data", "sample_eval_doc.pdf")

    if not os.path.exists(sample_pdf):
        print(f"❌ Error: Benchmark PDF missing at '{sample_pdf}'. Please add a sample PDF for CI.")
        sys.exit(1)

    print(f"📄 Indexing benchmark PDF: {sample_pdf}...")
    num_chunks = rag.process_and_reset_pdf(sample_pdf, "sample_eval_doc.pdf")
    print(f"✓ Indexed {num_chunks} chunks successfully.\n")

    # 2. Load Evaluation Dataset
    eval_dataset_path = os.path.join("data", "eval_dataset.json")
    with open(eval_dataset_path, "r") as f:
        eval_cases = json.load(f)

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    faithfulness_scores = []
    relevancy_scores = []

    print(f"🧪 Running {len(eval_cases)} evaluation cases through LLM Judge ({JUDGE_MODEL})...\n")

    for idx, case in enumerate(eval_cases, 1):
        query = case["query"]
        category = case.get("category", "General")

        # Run RAG answer
        result = rag.answer(query=query)
        answer = result["answer"]
        contexts = "\n".join(result["retrieved_contexts"])

        # Judge evaluation
        eval_result = evaluate_with_judge(groq_client, query, contexts, answer)
        
        f_score = float(eval_result.get("faithfulness", 0.0))
        r_score = float(eval_result.get("relevancy", 0.0))

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)

        print(f"[{idx}/{len(eval_cases)}] Category: {category}")
        print(f"  Q: {query}")
        print(f"  A: {answer[:80]}...")
        print(f"  Score -> Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}")
        print(f"  Reason: {eval_result.get('reasoning')}\n")
        
        time.sleep(0.5)  # Rate limit safety

    # 3. Calculate Averages & Validate Quality Thresholds
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)

    print("=" * 60)
    print("📊 FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Average Faithfulness Score: {avg_faithfulness:.2f} (Target: >= {FAITHFULNESS_THRESHOLD})")
    print(f"Average Relevancy Score   : {avg_relevancy:.2f} (Target: >= {RELEVANCY_THRESHOLD})")

    # Save artifact report
    report = {
        "avg_faithfulness": avg_faithfulness,
        "avg_relevancy": avg_relevancy,
        "threshold": FAITHFULNESS_THRESHOLD,
        "total_cases": len(eval_cases),
        "status": "PASSED" if avg_faithfulness >= FAITHFULNESS_THRESHOLD and avg_relevancy >= RELEVANCY_THRESHOLD else "FAILED"
    }
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # 4. Pass/Fail Gate Check
    if avg_faithfulness >= FAITHFULNESS_THRESHOLD and avg_relevancy >= RELEVANCY_THRESHOLD:
        print("\n✅ QUALITY GATE PASSED: PDF RAG Pipeline is performing reliably!")
        sys.exit(0)
    else:
        print("\n❌ QUALITY GATE FAILED: Performance dropped below strict 0.80 quality threshold.")
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()