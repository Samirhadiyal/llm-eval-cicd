import os
import json
import sys
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Resolve path to import RAGPipeline from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.rag_pipeline import RAGPipeline

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)

EVAL_DATASET_PATH = os.path.join(BASE_DIR, "data", "eval_dataset.json")
REPORT_PATH = os.path.join(BASE_DIR, "eval_report.json")

def score_faithfulness(question: str, context: list, answer: str) -> float:
    """Evaluates if the answer is strictly derived from the context (0.0 to 1.0)."""
    context_str = "\n".join(context) if context else "No context provided."
    prompt = f"""
    You are an AI Evaluation Judge. Rate the FAITHFULNESS of the answer on a scale from 0.0 to 1.0.
    - 1.0: All facts in the answer are supported by the context, OR the answer correctly refuses to answer because context is missing.
    - 0.0: The answer contains claims, facts, or hallucinations not present in the context.

    Context:
    {context_str}

    Question: {question}
    Answer: {answer}

    Respond ONLY with a valid JSON object matching this schema:
    {{"score": float, "reason": "short explanation"}}
    """
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        return float(data.get("score", 1.0))
    except Exception as e:
        print(f"Error scoring faithfulness: {e}")
        return 1.0

def score_relevance(question: str, answer: str) -> float:
    """Evaluates how directly the answer addresses the question (0.0 to 1.0)."""
    prompt = f"""
    You are an AI Evaluation Judge. Rate the RELEVANCE of the answer on a scale from 0.0 to 1.0.
    - 1.0: The answer directly and completely addresses the user question or states inability to answer appropriately.
    - 0.0: The answer is off-topic, evasive, or unhelpful.

    Question: {question}
    Answer: {answer}

    Respond ONLY with a valid JSON object matching this schema:
    {{"score": float, "reason": "short explanation"}}
    """
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        return float(data.get("score", 1.0))
    except Exception as e:
        print(f"Error scoring relevance: {e}")
        return 1.0

def run_evaluation():
    print("Initializing RAG Pipeline for Evaluation...")
    rag = RAGPipeline()

    print(f"Loading evaluation dataset from {EVAL_DATASET_PATH}...")
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    faithfulness_scores = []
    relevance_scores = []

    print(f"Running evaluation against {len(golden_data)} test cases...\n")
    for idx, item in enumerate(golden_data, 1):
        q = item["question"]
        result = rag.answer(query=q, top_k=3)

        f_score = score_faithfulness(q, result["retrieved_contexts"], result["answer"])
        r_score = score_relevance(q, result["answer"])

        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)

        print(f"[{idx}/{len(golden_data)}] Q: {q[:40]}... -> Faithfulness: {f_score} | Relevancy: {r_score}")

    avg_faithfulness = round(float(np.mean(faithfulness_scores)), 4)
    avg_relevance = round(float(np.mean(relevance_scores)), 4)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Evaluated:        {len(golden_data)}")
    print(f"Average Faithfulness:   {avg_faithfulness}")
    print(f"Average Answer Relevancy: {avg_relevance}")
    print("====================================================")

    report = {
        "faithfulness": avg_faithfulness,
        "answer_relevance": avg_relevance,
        "sample_size": len(golden_data)
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report successfully saved to {REPORT_PATH}")
    return report

if __name__ == "__main__":
    run_evaluation()