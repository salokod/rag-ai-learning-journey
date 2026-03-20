# Module 12: Observability with Langfuse

## Goal
Set up Langfuse -- the open-source alternative to Galileo -- to see everything your LLM system is doing. Trace calls, manage prompts, track quality over time.

---

## Why Observability?

Think about it this way. You have a CNC machine on your floor. It has gauges, readouts, alarm lights. You can see spindle speed, feed rate, coolant temp, tool wear -- all in real time. You would never run that machine blind.

But right now, your LLM pipeline? It is a black box. Something goes in, something comes out. If the output is bad, you have no idea where it went wrong. Was it the retrieval? The prompt? The model having a bad day?

Langfuse is the dashboard for your LLM system. It gives you the gauges.

---

## Setting Up Langfuse

You have two options. Pick whichever works for you.

### Option A: Langfuse Cloud (quickest -- 2 minutes)

1. Go to https://cloud.langfuse.com and sign up (free tier, no credit card)
2. Create a new project -- call it something like "manufacturing-rag"
3. Go to Settings > API Keys and create a key pair

You will get two keys. Add them to your `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Option B: Self-Hosted with Docker (full control)

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Open http://localhost:3000, create an account, create a project, grab your keys. Same `.env` format but with:

```
LANGFUSE_HOST=http://localhost:3000
```

### Install the SDK

```bash
pip install langfuse
```

Let's verify the connection works. Try this:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

langfuse = Langfuse()
print(langfuse.auth_check())
```

Run it:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from langfuse import Langfuse
print(Langfuse().auth_check())
"
```

You should see `True`. If you get an error, double-check your `.env` keys.

---

## Exercise 1: Your First Traced Function

Langfuse SDK v3 (released June 2025) uses the `@observe` decorator. This is the simplest way to trace -- you just decorate a function and Langfuse captures everything automatically.

Let's start with something dead simple.

```python
# 12-observability-with-langfuse/ex1_first_trace.py
"""Your first Langfuse trace -- one decorator, one function."""

from dotenv import load_dotenv
load_dotenv()

from langfuse.decorators import observe
import ollama
```

That is the setup. Now let's write one function and decorate it:

```python
@observe()
def describe_task(task_name: str) -> str:
    """Generate a short task description."""
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "Write a 2-sentence manufacturing task description."},
            {"role": "user", "content": task_name},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]
```

That is it. The `@observe()` decorator tells Langfuse: "trace this function -- capture its name, inputs, outputs, and timing."

Now call it:

```python
result = describe_task("Torque check on Frame #4200 bolts")
print(result)
```

Run the file. You will see the LLM output in your terminal like normal. But now go open your Langfuse dashboard.

**Go to your dashboard now. See that trace?**

That is your function call. Click on it. You will see:
- The function name (`describe_task`)
- The input (`"Torque check on Frame #4200 bolts"`)
- The output (the full generated text)
- How long it took
- A timestamp

One decorator. That is all it took.

Let's call it a few more times so you have more data to look at:

```python
tasks = [
    "Inspect weld joints on Assembly A",
    "Replace hydraulic seals on 200-ton press",
    "Calibrate digital caliper per SOP-CAL-003",
]

for task in tasks:
    result = describe_task(task)
    print(f"\n--- {task} ---")
    print(result)
```

Here is the complete file:

```python
# 12-observability-with-langfuse/ex1_first_trace.py
"""Your first Langfuse trace -- one decorator, one function."""

from dotenv import load_dotenv
load_dotenv()

from langfuse.decorators import observe
import ollama


@observe()
def describe_task(task_name: str) -> str:
    """Generate a short task description."""
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "Write a 2-sentence manufacturing task description."},
            {"role": "user", "content": task_name},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


# Try it once
result = describe_task("Torque check on Frame #4200 bolts")
print(result)

# Now a few more
tasks = [
    "Inspect weld joints on Assembly A",
    "Replace hydraulic seals on 200-ton press",
    "Calibrate digital caliper per SOP-CAL-003",
]

for task in tasks:
    result = describe_task(task)
    print(f"\n--- {task} ---")
    print(result)

print("\n--- Check your Langfuse dashboard. You should see 4 traces. ---")
```

---

## Exercise 2: Tracing a RAG Pipeline with Nested Spans

One function is nice. But your RAG pipeline has multiple steps: retrieve documents, build a prompt, call the LLM. You want to see each step separately.

Here is the trick: when you nest `@observe()` functions inside each other, Langfuse automatically creates parent-child spans. Let's build this up piece by piece.

First, set up a small knowledge base:

```python
# 12-observability-with-langfuse/ex2_rag_tracing.py
"""Trace a full RAG pipeline -- retrieval, generation, evaluation."""

from dotenv import load_dotenv
load_dotenv()

from langfuse.decorators import observe
import chromadb
import ollama

# Quick knowledge base
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
```

Now let's write three separate traced functions -- one for each pipeline step:

```python
@observe()
def retrieve_docs(question: str, n_results: int = 2) -> list:
    """Retrieve relevant documents from ChromaDB."""
    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0]
```

Notice how this is just a normal function with `@observe()` on top. Langfuse will capture the input (`question`), the output (the list of docs), and the timing.

Next, the generation step:

```python
@observe()
def generate_answer(question: str, context_docs: list) -> str:
    """Generate an answer using retrieved context."""
    context_str = "\n".join(context_docs)
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "Answer using ONLY the provided context. Cite document IDs."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]
```

And a quick evaluation step:

```python
@observe()
def evaluate_answer(answer: str, context_docs: list) -> dict:
    """Quick heuristic evaluation of the answer."""
    scores = {
        "length_ok": 10 <= len(answer.split()) <= 200,
        "uses_context": any(doc[:20] in answer for doc in context_docs),
        "not_empty": len(answer.strip()) > 0,
    }
    scores["overall"] = sum(scores.values()) / len(scores)
    return scores
```

Now the key part. We write one parent function that calls all three:

```python
@observe()
def rag_query(question: str) -> dict:
    """Full RAG pipeline -- retrieve, generate, evaluate."""
    docs = retrieve_docs(question)
    answer = generate_answer(question, docs)
    scores = evaluate_answer(answer, docs)
    return {"answer": answer, "scores": scores}
```

**This is where it gets good.** Because `rag_query` is decorated with `@observe()`, and it calls three other `@observe()` functions, Langfuse will show the parent trace with three child spans nested inside it. Just like a flame graph in traditional performance profiling.

Run some queries:

```python
questions = [
    "What is the torque spec for M10 bolts?",
    "What form do I use for visual inspection?",
    "What is the maximum interpass temperature for welding?",
]

for q in questions:
    result = rag_query(q)
    print(f"Q: {q}")
    print(f"A: {result['answer'][:80]}...")
    print(f"Scores: {result['scores']}")
    print()

print("--- Open Langfuse. Click on a trace. See the nested spans? ---")
print("--- You can see exactly how long retrieval vs generation took. ---")
```

Go check your dashboard now. Click on one of the `rag_query` traces. You will see:

```
rag_query (total: 2.1s)
  +-- retrieve_docs (0.05s)
  +-- generate_answer (1.9s)
  +-- evaluate_answer (0.01s)
```

Notice how generation dominates the time? That is the kind of insight you get for free. On a factory floor, this is like seeing which station in your production line is the bottleneck.

---

## Exercise 3: Prompt Management

Here is a real-world scenario. Your team has been tweaking the system prompt for weeks. Someone changes it, quality drops, nobody knows what the prompt was last Tuesday. Sound familiar? It is the manufacturing equivalent of someone adjusting machine settings without logging it.

Langfuse lets you store and version prompts in the dashboard, then fetch them in code. The prompt lives in Langfuse, not in your Python file.

```python
# 12-observability-with-langfuse/ex3_prompt_management.py
"""Manage prompts through Langfuse -- version control without code deploys."""

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from langfuse.decorators import observe
import ollama

langfuse = Langfuse()
```

First, let's create a prompt in Langfuse:

```python
langfuse.create_prompt(
    name="task-description-generator",
    prompt="""You are a manufacturing technical writer for an ISO 9001 facility.

Write a task description for: {{task_name}}
Context: {{context}}

Rules:
- 3-5 numbered steps starting with action verbs
- Include PPE/safety requirements
- Reference specific forms and specs
- Active voice, 50-100 words""",
    labels=["production"],
)
print("Prompt created in Langfuse.")
```

Run that once. Now go to your Langfuse dashboard and look under Prompts. You will see "task-description-generator" with version 1 and the label "production."

Now let's use it in code:

```python
@observe()
def generate_from_managed_prompt(task_name: str, context: str) -> str:
    """Fetch the prompt from Langfuse and use it."""
    # Fetch the production version of the prompt
    prompt = langfuse.get_prompt("task-description-generator", label="production")

    # Compile it -- this fills in the {{variables}}
    compiled = prompt.compile(task_name=task_name, context=context)

    response = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": compiled}],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


result = generate_from_managed_prompt(
    task_name="Inspect weld joints on Frame Assembly A",
    context="AWS D1.1, Form QC-107, fillet gauge required",
)
print(result)
```

**Why does this matter?** Because now you can:

1. Change the prompt in the Langfuse UI -- no code deploy needed
2. See every version of the prompt and when it changed
3. A/B test prompt versions
4. Roll back instantly if a new prompt tanks quality

It is like having a controlled recipe book for your LLM. Nobody can secretly tweak the recipe without it being logged.

Try changing the prompt in the Langfuse UI (add "Include lockout/tagout where applicable" to the rules). Then re-run the code. The new prompt gets fetched automatically.

---

## Exercise 4: Adding Quality Scores to Traces

You are generating task descriptions and tracing them. But are they any good? Let's attach quality scores directly to traces so you can track quality over time.

```python
# 12-observability-with-langfuse/ex4_quality_scores.py
"""Attach quality scores to traces for monitoring over time."""

from dotenv import load_dotenv
load_dotenv()

from langfuse.decorators import observe, langfuse_context
import ollama
import re
```

First, a scoring function:

```python
def score_task_description(text: str) -> dict:
    """Score a task description on key quality dimensions."""
    scores = {}
    scores["has_numbered_steps"] = len(re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)) >= 3
    scores["has_safety"] = any(w in text.lower() for w in ["ppe", "safety", "lockout", "gloves", "helmet"])
    scores["has_references"] = bool(re.search(r'[A-Z]{2,}-\d{2,}', text))
    scores["good_length"] = 30 <= len(text.split()) <= 150
    scores["starts_with_verb"] = bool(re.search(r'^\s*\d+[\.\)]\s*[A-Z][a-z]+\b', text, re.MULTILINE))
    scores["overall"] = sum(scores.values()) / len(scores)
    return scores
```

Now, here is the key part. Inside an `@observe()` function, you can use `langfuse_context` to attach scores to the current trace:

```python
@observe()
def generate_and_score(task_name: str) -> dict:
    """Generate a task description and score it."""
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "Write a manufacturing task description with numbered steps, safety notes, and spec references."},
            {"role": "user", "content": f"Task: {task_name}"},
        ],
        options={"temperature": 0.0},
    )
    answer = response["message"]["content"]

    # Score it
    scores = score_task_description(answer)

    # Attach scores to the current trace in Langfuse
    for name, value in scores.items():
        langfuse_context.score_current_trace(
            name=name,
            value=float(value),
        )

    return {"answer": answer, "scores": scores}
```

Let's run it on several tasks:

```python
tasks = [
    "Inspect weld joints on Frame A",
    "Set up CNC for shaft machining",
    "Perform daily forklift inspection",
    "Calibrate pressure gauge",
    "Replace conveyor belt rollers",
]

print("=== Generating and Scoring ===\n")
for task in tasks:
    result = generate_and_score(task)
    status = "PASS" if result["scores"]["overall"] >= 0.6 else "FAIL"
    print(f"[{status}] {task}")
    print(f"  Overall: {result['scores']['overall']:.0%}")
    for k, v in result["scores"].items():
        if k != "overall":
            print(f"    {k}: {'yes' if v else 'no'}")
    print()

print("--- Go to Langfuse. Click on Scores in the sidebar. ---")
print("--- You can filter traces by score value. ---")
print("--- Find the ones that failed and see why. ---")
```

**Now imagine this running for a week.** Every time your system generates a task description, the score gets logged. In Langfuse you can:

- See quality trending over time (is it getting better or worse?)
- Filter to just the low-scoring outputs (what went wrong?)
- Correlate quality drops with prompt changes (did someone break something?)

This is your SPC chart for LLM output quality. Just like statistical process control catches a CNC machine drifting out of tolerance, Langfuse catches your LLM drifting out of quality specs.

---

## Exercise 5: Putting It All Together -- A Monitored RAG Function

Let's combine everything: tracing, nested spans, prompt management, and quality scores in one clean pipeline.

```python
# 12-observability-with-langfuse/ex5_monitored_rag.py
"""Complete monitored RAG pipeline using @observe decorator throughout."""

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import chromadb
import ollama
import re

langfuse = Langfuse()

# --- Knowledge Base Setup ---
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="monitored_kb")
collection.add(
    ids=["MT-302", "QC-107", "WPS-201", "SOP-FL-001", "CAL-003"],
    documents=[
        "Torque Spec MT-302: M8=25-30Nm, M10=45-55Nm for Frame #4200",
        "Form QC-107: Visual inspection checklist, requires inspector badge and date",
        "WPS-201: GMAW welding, 75/25 Ar/CO2, interpass temp 400F max",
        "SOP-FL-001: Daily forklift inspection, check tires/horn/lights/brakes before each shift",
        "CAL-003: Caliper calibration using NIST-traceable gauge blocks at 0.5/1.0/2.0/4.0 inches",
    ],
)


# --- Pipeline Steps (each traced separately) ---

@observe()
def retrieve(question: str) -> dict:
    """Retrieve relevant documents."""
    results = collection.query(query_texts=[question], n_results=2)
    return {
        "documents": results["documents"][0],
        "ids": results["ids"][0],
    }


@observe()
def generate(question: str, context_docs: list) -> str:
    """Generate answer from context."""
    # Try to fetch managed prompt, fall back to local
    try:
        prompt_obj = langfuse.get_prompt("task-description-generator", label="production")
        system_msg = "You are a manufacturing technical writer. Answer using only the provided context."
    except Exception:
        system_msg = "You are a manufacturing technical writer. Answer using only the provided context."

    context_str = "\n".join(context_docs)
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


@observe()
def evaluate(answer: str, source_ids: list) -> dict:
    """Evaluate the answer quality."""
    scores = {
        "has_steps": len(re.findall(r'^\s*\d+[\.\)]', answer, re.MULTILINE)) >= 2,
        "has_safety": any(w in answer.lower() for w in ["ppe", "safety", "lockout", "gloves"]),
        "has_refs": bool(re.search(r'[A-Z]{2,}-\d{2,}', answer)),
        "cites_source": any(sid in answer for sid in source_ids),
        "good_length": 20 <= len(answer.split()) <= 200,
    }
    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


# --- Main Pipeline ---

@observe()
def monitored_rag_query(question: str) -> dict:
    """Full monitored RAG pipeline."""
    # Step 1: Retrieve
    retrieved = retrieve(question)

    # Step 2: Generate
    answer = generate(question, retrieved["documents"])

    # Step 3: Evaluate
    scores = evaluate(answer, retrieved["ids"])

    # Attach scores to the trace
    for name, value in scores.items():
        langfuse_context.score_current_trace(
            name=name,
            value=float(value),
        )

    # Tag the trace with metadata
    langfuse_context.update_current_trace(
        metadata={
            "source_ids": retrieved["ids"],
            "model": "qwen3:8b",
            "pipeline_version": "v1",
        }
    )

    return {
        "answer": answer,
        "sources": retrieved["ids"],
        "scores": scores,
    }


# --- Run It ---

questions = [
    "What is the torque spec for M10 bolts on Frame #4200?",
    "What do I need for a visual inspection?",
    "What gas mix does WPS-201 require for GMAW welding?",
    "How do I do a daily forklift inspection?",
    "What gauge blocks do I need for caliper calibration?",
]

print("=== Monitored RAG Pipeline ===\n")
for q in questions:
    result = monitored_rag_query(q)
    status = "PASS" if result["scores"]["overall"] >= 0.6 else "FAIL"
    print(f"[{status}] Q: {q}")
    print(f"  A: {result['answer'][:80]}...")
    print(f"  Sources: {result['sources']}")
    print(f"  Overall: {result['scores']['overall']:.0%}")
    print()

print("=== What to Check in Langfuse ===")
print("1. Click on any trace -- see the retrieve/generate/evaluate spans")
print("2. Click Scores in the sidebar -- see quality over all queries")
print("3. Filter by score < 0.6 to find the weak outputs")
print("4. Look at the timing -- is retrieval or generation the bottleneck?")
print("5. Check the metadata tab -- see model version and source IDs")
```

---

## Takeaways

1. **Langfuse is your open-source Galileo** -- free, self-hosted, full tracing and scoring
2. **The `@observe()` decorator is all you need** -- one line per function, Langfuse handles the rest
3. **Nested `@observe()` functions create span trees** -- see exactly where time is spent
4. **Prompt management** means prompt changes are tracked and reversible, just like version-controlled machine settings
5. **Quality scores on traces** let you build an SPC chart for your LLM -- catch drift before it becomes a problem

## What's Next

You can observe and score your pipeline now. But your evaluation is only as good as your test data. Module 13 teaches you to build golden datasets and regression benchmarks -- the test infrastructure that makes all this monitoring meaningful.
