import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

def build_pdf():
    os.makedirs("data", exist_ok=True)
    pdf_path = "data/sample_eval_doc.pdf"
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=12
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=14,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    story = []

    # PAGE 1: System Overview & Architecture
    story.append(Paragraph("Enterprise RAG System Architecture & Evaluation Benchmark", title_style))
    story.append(Paragraph("Document Version: 2.1 | Deployment Pipeline Specification", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. System Overview", section_style))
    story.append(Paragraph(
        "This specification document outlines the production architecture for the Automated RAG Evaluation "
        "and Quality Gate Service. The system provides high-throughput document retrieval, cross-encoder reranking, "
        "and automated CI/CD validation for large language model applications.",
        body_style
    ))

    story.append(Paragraph("2. Technical Stack & Components", section_style))
    story.append(Paragraph("The core pipeline consists of the following key frameworks and infrastructure:", body_style))
    
    table_data = [
        ["Component", "Technology", "Primary Purpose"],
        ["Vector Database", "ChromaDB", "Dense embedding storage and similarity search"],
        ["Sparse Search", "BM25 (rank-bm25)", "Exact keyword matching and token scoring"],
        ["Fusion Engine", "Reciprocal Rank Fusion", "Blending dense and sparse retrieval ranks"],
        ["Reranker Model", "FlashRank Cross-Encoder", "High-precision top-N context re-scoring"],
        ["LLM Provider", "Groq Cloud API", "Ultra-low latency inference using Llama models"],
        ["CI/CD Engine", "GitHub Actions", "Automated quality gate testing on code push"]
    ]
    
    t = Table(table_data, colWidths=[120, 150, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#334155')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("3. Operational Requirements & Quality Gates", section_style))
    story.append(Paragraph(
        "To guarantee zero regression, all pull requests must pass an automated CI Quality Gate. "
        "The automated judge measures Faithfulness and Relevancy against a minimum threshold of 0.80. "
        "System deployments are automatically aborted if evaluation scores fall below this target threshold.",
        body_style
    ))

    # PAGE 2: Guardrails, Performance Benchmarks & SLAs
    story.append(PageBreak())
    story.append(Paragraph("4. Guardrails & Refusal Policies", section_style))
    story.append(Paragraph(
        "The RAG system enforces strict out-of-domain guardrails. When queries fall outside the context of the "
        "active document knowledge base, the system strictly outputs standard refusal messages: 'I cannot answer this "
        "based on the available documentation.' This prevents false hallucinations in mission-critical environments.",
        body_style
    ))

    story.append(Paragraph("5. Performance Benchmarks & SLA Targets", section_style))
    story.append(Paragraph("Expected system latencies across production deployment stages:", body_style))

    sla_data = [
        ["Pipeline Stage", "Target SLA", "Model / Method Used"],
        ["Hybrid Retrieval", "< 50 ms", "ChromaDB + BM25Okapi"],
        ["Cross-Encoder Rerank", "< 100 ms", "ms-marco-MiniLM-L-12-v2"],
        ["LLM Generation", "< 1200 ms", "Groq llama-3.1-8b-instant"],
        ["Total End-to-End SLA", "< 1500 ms", "Full Pipeline Execution"]
    ]

    t2 = Table(sla_data, colWidths=[150, 120, 250])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#334155')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
    ]))
    story.append(t2)

    doc.build(story)
    print("✅ Successfully generated multi-page tailored benchmark PDF at data/sample_eval_doc.pdf!")

if __name__ == "__main__":
    build_pdf()