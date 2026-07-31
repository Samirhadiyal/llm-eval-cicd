# 🛡️ LLM Eval CI/CD Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-ff6600?style=for-the-badge&logo=databricks&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-f34f29?style=for-the-badge&logo=speedtest&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

**A production-grade, zero-cost MLOps pipeline that brings software engineering rigor (unit testing & automated CI/CD guardrails) to Retrieval-Augmented Generation (RAG) applications.**


</div>

---

## 🎯 Interactive Demo

<div align="center">

![RAG System Demo](media/demo.gif)

*Side-by-side view: Interactive RAG Frontend (Right) & Automated CI/CD Regression Guardrail on GitHub Actions (Left).*

</div>

---

## 🚨 What Problem Does This Solve?

Traditional software fails **noisily**—throwing HTTP 500 errors or crashing when code breaks. 

LLM applications (like RAG systems) fail **silently**. When a developer tweaks a prompt template, updates an embedding model, or alters vector chunk sizes, the app often returns **confident, believable hallucinations** while still responding with `HTTP 200 OK`.

### Core Engineering Challenges Solved:
1. **Unnoticed Quality Regressions:** Small codebase changes can silently break responses for dozens of edge cases.
2. **Hallucination & Data Leakage Risks:** Users asking out-of-domain or false-premise queries can trick unchecked LLMs into making up facts.
3. **Subjective Manual QA:** Developers waste hours manually spot-checking chat responses without quantitative metrics.

### The Solution:
This project treats **LLM evaluation as automated continuous integration**. Every code push automatically runs a **35-pair golden benchmark dataset** through an **LLM-as-a-Judge** framework (`llama-3.3-70b-versatile`). If quality scores 


<section class="tech-stack-section">
  <h2>🛠️ Tech Stack</h2>
  <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr style="background-color: #f2f2f2;">
        <th>Component</th>
        <th>Technology</th>
        <th>Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>API Framework</strong></td>
        <td>FastAPI</td>
        <td>High-performance asynchronous REST API backend &amp; static UI serving</td>
      </tr>
      <tr>
        <td><strong>Vector DB</strong></td>
        <td>ChromaDB</td>
        <td>Local persistent vector storage for dense embeddings</td>
      </tr>
      <tr>
        <td><strong>Embeddings</strong></td>
        <td>HuggingFace (all-MiniLM-L6-v2)</td>
        <td>384-dimensional dense vector embeddings</td>
      </tr>
      <tr>
        <td><strong>Inference Engine</strong></td>
        <td>Groq API (llama-3.1-8b-instant)</td>
        <td>Low-latency LLM inference engine for RAG generation</td>
      </tr>
      <tr>
        <td><strong>LLM Judge</strong></td>
        <td>Groq API (llama-3.3-70b-versatile)</td>
        <td>Automated evaluation judge for Faithfulness &amp; Relevancy</td>
      </tr>
      <tr>
        <td><strong>CI/CD Orchestration</strong></td>
        <td>GitHub Actions</td>
        <td>Automated continuous evaluation pipeline triggered on code pushes</td>
      </tr>
      <tr>
        <td><strong>Frontend UI</strong></td>
        <td>HTML5 / Tailwind CSS / JS</td>
        <td>Lightweight, single-process interactive web dashboard</td>
      </tr>
    </tbody>
  </table>
</section>

<hr>

<section class="ci-cd-section">
  <h2>🚦 CI/CD Quality Gate &amp; Metrics</h2>
  <p>
    Every GitHub push triggers <code>.github/workflows/eval_ci.yml</code> which executes <code>tests/eval_runner.py</code>. The evaluation runner tests 35 benchmark pairs across four query categories:
  </p>
  <ul>
    <li><strong>Factual Lookups:</strong> Verifies precise retrieval of operational rules and SLAs.</li>
    <li><strong>Out-of-Domain Guardrails:</strong> Verifies that missing context produces a clean refusal instead of hallucinations.</li>
    <li><strong>False Premise Correction:</strong> Verifies that erroneous assumptions in queries are corrected using document facts.</li>
    <li><strong>Adversarial Directives:</strong> Ensures prompt-injection attempts do not breach context constraints.</li>
  </ul>

  <h3>Evaluated Metrics:</h3>
  <ul>
    <li><strong>Faithfulness (0.0 - 1.0):</strong> Measures whether the generated answer is mathematically grounded only in the retrieved ChromaDB chunks.</li>
    <li><strong>Answer Relevancy (0.0 - 1.0):</strong> Measures how directly the generated response addresses the user's input prompt.</li>
  </ul>

  <pre><code>CI Check Status:
  - Faithfulness Target : &gt;= 0.80
  - Relevancy Target    : &gt;= 0.80

[SUCCESS] Quality Gate Passed. Artifact 'eval-report.json' uploaded.</code></pre>
</section>

<hr>

<section class="getting-started-section">
  <h2>🚀 Getting Started</h2>
  
  <h3>Prerequisites</h3>
  <ul>
    <li>Python 3.11+</li>
    <li>A free Groq API Key (<a href="https://console.groq.com/" target="_blank" rel="noopener noreferrer">Get one here</a>)</li>
  </ul>

  <h3>Installation &amp; Local Setup</h3>
  
  <ol>
    <li>
      <p><strong>Clone the Repository:</strong></p>
      <pre><code>git clone https://github.com/Samirhadiyal/llm-eval-cicd.git
cd llm-eval-cicd</code></pre>
    </li>
      <p><strong>Ingest Documents into ChromaDB:</strong></p>
      <pre><code>python src/ingest.py</code></pre>
    </li>
    <li>
      <p><strong>Run the FastAPI Web Application:</strong></p>
      <pre><code>uvicorn src.app:app --reload</code></pre>
      <ul>
        <li><strong>Web Interface:</strong> Access the UI at <a href="http://127.0.0.1:8000/" target="_blank">http://127.0.0.1:8000/</a></li>
        <li><strong>Interactive API Docs:</strong> Access Swagger at <a href="http://127.0.0.1:8000/docs" target="_blank">http://127.0.0.1:8000/docs</a></li>
      </ul>
    </li>
    <li>
      <p><strong>Execute the Evaluation Suite Locally:</strong></p>
      <pre><code>python tests/eval_runner.py</code></pre>
    </li>
  </ol>
</section>for **Faithfulness** or **Answer Relevancy** drop below **`0.80`**, the CI/CD pipeline turns red and blocks the deployment.

