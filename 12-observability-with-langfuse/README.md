# Module 12: Observability with Langfuse

## Goal
Set up production-grade LLM observability using Langfuse — the open-source alternative to Galileo. Track every LLM call, manage prompts, debug failures, and monitor quality over time.

---

## Concepts

### What Is LLM Observability?

In traditional software, you have logs, metrics, and traces. LLM applications need the same, plus:
- **What prompt was sent?** (exact text, version)
- **What context was retrieved?** (RAG documents)
- **What did the model output?** (the full response)
- **How long did it take?** (latency breakdown)
- **How much did it cost?** (token counts)
- **Was the output good?** (quality scores)

### Langfuse = Open-Source Galileo

| Feature | Langfuse | Galileo |
|---------|----------|---------|
| **License** | MIT (open source) | Proprietary |
| **Self-hosted** | Yes (Docker) | No (cloud only) |
| **Tracing** | Full LLM call tracing | Full tracing |
| **Prompt management** | Version-controlled prompts | Prompt lab |
| **Evaluation** | Integrates with Ragas/DeepEval | Built-in evaluation |
| **Cost** | Free (self-hosted) | Enterprise pricing |
| **Best for** | Teams wanting control | Enterprise with budget |

### What You'll Track

```
┌─────────────────────────────────────────────────┐
│                  LANGFUSE DASHBOARD              │
│                                                   │
│  Traces: 1,247 today                             │
│  Avg Latency: 2.3s                               │
│  Avg Quality Score: 0.84                         │
│  Cost: $0.00 (local Ollama)                      │
│                                                   │
│  ┌──────────────────────────────────┐            │
│  │ Trace: "task-desc-gen-001"       │            │
│  │  ├── Retrieve (0.1s)            │            │
│  │  │   └── 3 docs from ChromaDB   │            │
│  │  ├── Generate (1.8s)            │            │
│  │  │   └── llama3.1:8b, 245 tokens│            │
│  │  └── Evaluate (0.5s)            │            │
│  │      └── Quality: 0.87          │            │
│  └──────────────────────────────────┘            │
└─────────────────────────────────────────────────┘
```

---

## Environment Setup

### Option A: Langfuse Cloud (Quick Start)

1. Sign up at https://langfuse.com (free tier available)
2. Create a project, get your API keys
3. Add to your `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Option B: Self-Hosted (Full Control)

```bash
# Clone and run Langfuse locally with Docker
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d

# Langfuse UI will be at http://localhost:3000
# Create an account and project, get API keys
```

---

## Exercise 1: Tracing Your RAG Pipeline

```python
# 12-observability-with-langfuse/ex1_tracing.py
"""Add Langfuse tracing to your RAG pipeline."""

import os
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
import chromadb
import ollama

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-test"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-test"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

# Set up a simple knowledge base
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="traced_kb")
collection.add(
    ids=["MT-302", "QC-107", "WPS-201"],
    documents=[
        "Torque Spec MT-302: M8=25-30Nm, M10=45-55Nm for Frame #4200",
        "Form QC-107: Visual inspection checklist, requires inspector badge and date",
        "WPS-201: GMAW welding, 75/25 Ar/CO2, interpass temp 400F max",
    ],
)


def traced_rag_query(question: str) -> dict:
    """RAG pipeline with full Langfuse tracing."""

    # Create a trace for this entire request
    trace = langfuse.trace(
        name="rag-task-query",
        input={"question": question},
        metadata={"pipeline_version": "v1", "model": "llama3.1:8b"},
    )

    # SPAN 1: Retrieval
    retrieval_span = trace.span(
        name="retrieval",
        input={"query": question, "n_results": 2},
    )

    results = collection.query(query_texts=[question], n_results=2)
    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]

    retrieval_span.end(
        output={"doc_ids": retrieved_ids, "doc_count": len(retrieved_docs)},
    )

    # SPAN 2: Generation
    generation = trace.generation(
        name="llm-generation",
        model="llama3.1:8b",
        input={
            "system": "Answer using only the provided context.",
            "context": retrieved_docs,
            "question": question,
        },
        model_parameters={"temperature": 0.0},
    )

    context_str = "\n".join(retrieved_docs)
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "Answer using ONLY the provided context. Cite sources."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )

    answer = response["message"]["content"]
    generation.end(output={"answer": answer})

    # SPAN 3: Evaluation (quick heuristic)
    eval_span = trace.span(name="evaluation")
    quality_score = min(len(answer.split()) / 50, 1.0)  # Simple proxy
    has_citation = any(ref_id in answer for ref_id in retrieved_ids)

    # Log the evaluation score
    trace.score(name="quality", value=quality_score)
    trace.score(name="has_citation", value=1.0 if has_citation else 0.0)

    eval_span.end(
        output={"quality_score": quality_score, "has_citation": has_citation},
    )

    # Complete the trace
    trace.update(output={"answer": answer, "sources": retrieved_ids})

    return {"answer": answer, "sources": retrieved_ids, "quality": quality_score}


# Run some traced queries
test_questions = [
    "What is the torque spec for M10 bolts?",
    "What form do I use for visual inspection?",
    "What's the maximum interpass temperature for welding?",
]

print("=== Running Traced Queries ===\n")
for q in test_questions:
    result = traced_rag_query(q)
    print(f"Q: {q}")
    print(f"A: {result['answer'][:100]}...")
    print(f"Sources: {result['sources']}, Quality: {result['quality']:.2f}\n")

# Flush to ensure all traces are sent
langfuse.flush()

print("=== Check Your Langfuse Dashboard ===")
print("Open http://localhost:3000 (self-hosted) or https://cloud.langfuse.com")
print("You'll see:")
print("  - Each query as a trace with timing")
print("  - Retrieval, generation, and evaluation as spans")
print("  - Quality scores attached to each trace")
print("  - Full prompt and response text for debugging")
```

---

## Exercise 2: Prompt Management

```python
# 12-observability-with-langfuse/ex2_prompt_management.py
"""Use Langfuse to version-control and manage prompts."""

import os
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
import ollama

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-test"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-test"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

# Create/update a prompt in Langfuse
# This lets you change prompts WITHOUT changing code
try:
    langfuse.create_prompt(
        name="task-description-generator",
        prompt="""You are a manufacturing technical writer for an ISO 9001 facility.
Write task descriptions following these rules:
- 3-5 numbered steps starting with action verbs
- Include specific tool, spec, and form references
- Include PPE/safety requirements
- Active voice, 8th-grade reading level
- 50-100 words

Task: {{task_name}}
Context: {{context}}""",
        labels=["production"],
    )
    print("✓ Prompt created/updated in Langfuse")
except Exception as e:
    print(f"Note: {e}")
    print("(This is fine — Langfuse may not be running. Using local prompt.)")

# In production, you'd fetch the prompt from Langfuse:
def get_prompt_from_langfuse(name: str) -> str:
    """Fetch the latest production prompt from Langfuse."""
    try:
        prompt = langfuse.get_prompt(name, label="production")
        return prompt.compile(task_name="{{task_name}}", context="{{context}}")
    except Exception:
        # Fallback to local prompt
        return """You are a manufacturing technical writer.
Write 3-5 numbered steps starting with action verbs.
Include safety and reference requirements.

Task: {{task_name}}
Context: {{context}}"""

print("\n=== Prompt Management Benefits ===")
print("1. Version history: See every prompt change over time")
print("2. A/B testing: Run different prompt versions in production")
print("3. No code deploys: Change prompts without touching code")
print("4. Audit trail: Who changed what prompt, when, and why")
print("5. Rollback: Instantly revert to a previous prompt version")
```

---

## Exercise 3: Monitoring Quality Over Time

```python
# 12-observability-with-langfuse/ex3_quality_monitoring.py
"""Track quality metrics over time to catch degradation."""

import os
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
import ollama
import json
import re
from datetime import datetime

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-test"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-test"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)


def evaluate_and_trace(task_name: str, output: str, trace_id: str = None) -> dict:
    """Evaluate a task description and log metrics to Langfuse."""

    # Heuristic evaluation
    scores = {
        "format_compliance": 1.0 if re.findall(r'^\s*\d+[\.\)]', output, re.MULTILINE) else 0.0,
        "length_appropriate": 1.0 if 30 <= len(output.split()) <= 200 else 0.0,
        "has_safety": 1.0 if any(w in output.lower() for w in ["safety", "ppe", "lockout", "gloves"]) else 0.0,
        "has_references": 1.0 if re.search(r'[A-Z]{2,}-\d{2,}', output) else 0.0,
    }
    scores["overall"] = sum(scores.values()) / len(scores)

    # Log to Langfuse
    trace = langfuse.trace(
        name="quality-check",
        input={"task_name": task_name},
        output={"description": output},
        metadata={"timestamp": datetime.now().isoformat()},
    )

    for metric_name, value in scores.items():
        trace.score(name=metric_name, value=value)

    return scores


# Simulate monitoring over multiple generations
tasks = [
    "Inspect weld joints on Frame A",
    "Set up CNC for shaft machining",
    "Perform daily forklift inspection",
    "Calibrate pressure gauge",
    "Replace conveyor belt rollers",
]

print("=== Quality Monitoring Dashboard (simulated) ===\n")
all_scores = []

for task in tasks:
    output = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "Write a manufacturing task description with numbered steps, safety notes, and spec references."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        options={"temperature": 0.0},
    )["message"]["content"]

    scores = evaluate_and_trace(task, output)
    all_scores.append(scores)

    status = "✓" if scores["overall"] >= 0.75 else "⚠️" if scores["overall"] >= 0.5 else "✗"
    print(f"{status} {task}: {scores['overall']:.0%}")
    for k, v in scores.items():
        if k != "overall":
            print(f"    {k}: {'✓' if v else '✗'}")

# Aggregate stats
avg_overall = sum(s["overall"] for s in all_scores) / len(all_scores)
avg_safety = sum(s["has_safety"] for s in all_scores) / len(all_scores)
avg_format = sum(s["format_compliance"] for s in all_scores) / len(all_scores)

print(f"\n=== Aggregate Metrics ===")
print(f"Overall quality:    {avg_overall:.0%}")
print(f"Format compliance:  {avg_format:.0%}")
print(f"Safety inclusion:   {avg_safety:.0%}")

langfuse.flush()

print(f"\n=== In Langfuse Dashboard You'd See ===")
print("- Quality scores trending over time (daily/weekly)")
print("- Alerts when scores drop below threshold")
print("- Drill-down into specific low-scoring traces")
print("- Comparison between prompt versions")
print("- Cost per query and total spend tracking")
```

---

## Takeaways

1. **Langfuse is your open-source Galileo** — free, self-hosted, MIT-licensed
2. **Trace everything** — retrieval, generation, evaluation, all in one trace
3. **Prompt management** separates prompt changes from code deploys
4. **Quality monitoring** catches degradation before users notice
5. **Scores on traces** create a queryable quality history

## Setting the Stage for Module 13

You can evaluate and observe. But evaluation is only as good as your **test data**. Module 13 teaches you to build high-quality evaluation datasets, create golden test sets, and set up regression benchmarks that protect against quality degradation over time.
