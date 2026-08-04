import os
import sys
import json
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "llama-3.3-70b-versatile"

# Hand-annotated calibration set with known human ground truth (1.0 = Pass, 0.0 = Fail)
HUMAN_CALIBRATION_SET = [
    {
        "id": 1,
        "scenario": "Direct Factual Answer",
        "query": "What vector database is used for dense storage?",
        "context": "ChromaDB is used for dense embedding storage and similarity search.",
        "answer": "ChromaDB is used for dense embedding storage.",
        "human_faithfulness": 1.0,
        "human_relevancy": 1.0
    },
    {
        "id": 2,
        "scenario": "Hallucinated Answer",
        "query": "What vector database is used for dense storage?",
        "context": "ChromaDB is used for dense embedding storage and similarity search.",
        "answer": "The system uses Pinecone and PostgreSQL for vector embeddings.",
        "human_faithfulness": 0.0,
        "human_relevancy": 0.0
    },
    {
        "id": 3,
        "scenario": "Out-of-Domain Correct Refusal",
        "query": "What is the penalty for a no-ball in international cricket?",
        "context": "This document covers RAG system architecture and SLAs.",
        "answer": "I cannot answer this based on the available documentation.",
        "human_faithfulness": 1.0,
        "human_relevancy": 1.0
    },
    {
        "id": 4,
        "scenario": "Out-of-Domain Hallucination",
        "query": "What is the capital of France?",
        "context": "This document covers RAG system architecture and SLAs.",
        "answer": "The capital city of France is Paris.",
        "human_faithfulness": 0.0, # Not derived from context
        "human_relevancy": 0.0  # Should have refused
    },
    {
        "id": 5,
        "scenario": "False Premise Correct Refusal",
        "query": "Why does the document recommend PostgreSQL?",
        "context": "ChromaDB is used for dense embedding storage.",
        "answer": "I cannot answer this based on the available documentation.",
        "human_faithfulness": 1.0,
        "human_relevancy": 1.0
    }
]

def calibrate():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    judge_faithfulness = []
    judge_relevancy = []
    
    human_faithfulness = []
    human_relevancy = []

    print("=" * 65)
    print("⚖️ RUNNING LLM JUDGE CALIBRATION AGAINST HUMAN GROUND TRUTH")
    print("=" * 65)

    for case in HUMAN_CALIBRATION_SET:
        prompt = f"""
        You are an impartial MLOps evaluation judge.
        Scenario: {case['scenario']}
        User Query: {case['query']}
        Retrieved Context: {case['context']}
        Generated Answer: {case['answer']}
        
        Evaluate:
        1. Faithfulness (1.0 or 0.0): Strictly derived ONLY from context?
        2. Relevancy (1.0 or 0.0): Directly answers OR safely refuses out-of-domain/unanswerable queries?
        
        Return JSON ONLY: {{"faithfulness": 1.0 or 0.0, "relevancy": 1.0 or 0.0}}
        """
        
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        res = json.loads(response.choices[0].message.content)
        f_score = float(res.get("faithfulness", 0.0))
        r_score = float(res.get("relevancy", 0.0))

        judge_faithfulness.append(f_score)
        judge_relevancy.append(r_score)
        
        human_faithfulness.append(case["human_faithfulness"])
        human_relevancy.append(case["human_relevancy"])

        print(f"[{case['id']}/5] Scenario: {case['scenario']}")
        print(f"   Human Target -> F: {case['human_faithfulness']} | R: {case['human_relevancy']}")
        print(f"   Judge Score  -> F: {f_score} | R: {r_score}\n")

    # Calculate Alignment Percentage
    f_agreements = np.array(judge_faithfulness) == np.array(human_faithfulness)
    r_agreements = np.array(judge_relevancy) == np.array(human_relevancy)
    
    f_accuracy = np.mean(f_agreements) * 100
    r_accuracy = np.mean(r_agreements) * 100
    overall_accuracy = (f_accuracy + r_accuracy) / 2

    print("=" * 65)
    print("🎯 JUDGE ALIGNMENT RESULTS")
    print("=" * 65)
    print(f"Faithfulness Alignment : {f_accuracy:.1f}%")
    print(f"Relevancy Alignment    : {r_accuracy:.1f}%")
    print(f"Overall Judge Accuracy : {overall_accuracy:.1f}%")

    calibration_report = {
        "judge_model": JUDGE_MODEL,
        "faithfulness_alignment_pct": f_accuracy,
        "relevancy_alignment_pct": r_accuracy,
        "overall_alignment_pct": overall_accuracy,
        "status": "VALIDATED" if overall_accuracy >= 90.0 else "NEEDS_CALIBRATION"
    }

    report_path = os.path.join("data", "judge_calibration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2)

    print(f" Saved calibration report to '{report_path}'")

if __name__ == "__main__":
    calibrate()