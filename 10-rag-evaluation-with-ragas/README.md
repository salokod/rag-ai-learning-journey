# Module 10: RAG Evaluation with Ragas

## Goal
Master Ragas — the industry-standard open-source framework for evaluating RAG pipelines. Learn to measure faithfulness, relevancy, context quality, and answer correctness with research-backed metrics.

---

## Concepts

### What Is Ragas?

Ragas (Retrieval-Augmented Generation Assessment) is a framework specifically designed to evaluate RAG pipelines. It answers questions like:

- **Is the answer grounded in the retrieved context?** (Faithfulness)
- **Is the answer relevant to the question?** (Answer Relevancy)
- **Did we retrieve the right documents?** (Context Precision/Recall)

### The Ragas Metrics

| Metric | What It Measures | Score Range | Why You Care |
|--------|-----------------|-------------|-------------|
| **Faithfulness** | Is the answer supported by the context? | 0-1 | Catches hallucinations |
| **Answer Relevancy** | Does the answer address the question? | 0-1 | Catches off-topic responses |
| **Context Precision** | Are the retrieved docs actually useful? | 0-1 | Measures retrieval quality |
| **Context Recall** | Did we retrieve all needed info? | 0-1 | Catches missing context |
| **Answer Correctness** | Is the answer factually correct? | 0-1 | The ultimate quality check |

### Ragas Is Reference-Free (Mostly)

The killer feature: **Faithfulness and Answer Relevancy don't need ground-truth answers**. You can evaluate production RAG responses without pre-labeled data. Context Recall and Answer Correctness do need reference answers.

---

## Exercise 1: Your First Ragas Evaluation

```python
# 10-rag-evaluation-with-ragas/ex1_first_ragas_eval.py
"""Run your first Ragas evaluation on a RAG pipeline."""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Set up local models for Ragas (no API keys needed!)
llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0.0))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

# Create evaluation dataset from your RAG pipeline outputs
# In practice, you'd collect these from actual RAG queries
eval_data = {
    "question": [
        "What is the torque specification for M10 bolts on Frame Assembly #4200?",
        "What PPE is required for MIG welding?",
        "How do I perform lockout/tagout?",
    ],
    "answer": [
        "The torque specification for M10 bolts on Frame Assembly #4200 is 45-55 Nm, per specification MT-302.",
        "MIG welding requires an auto-darkening helmet (shade 10-13), leather welding gloves, FR clothing, and steel-toe boots.",
        "Lockout/tagout requires: 1) Notify affected operators, 2) Shut down using normal stop, 3) Isolate energy sources, 4) Apply personal lock and tag, 5) Release stored energy, 6) Verify zero energy state.",
    ],
    "contexts": [
        ["Torque Specification MT-302 for Frame Assembly #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Use calibrated torque wrench ±2%."],
        ["Standard PPE for welding: Auto-darkening helmet (shade 10-13), leather welding gloves, FR clothing, steel-toe boots, safety glasses under helmet."],
        ["LOTO Procedure: 1. Notify operators 2. Shut down machine 3. Isolate energy sources 4. Apply lock and tag 5. Release stored energy 6. Verify zero energy state."],
    ],
    "ground_truth": [
        "M10 bolts on Frame Assembly #4200 require 45-55 Nm torque per MT-302.",
        "Welding PPE: auto-darkening helmet shade 10-13, leather gloves, FR clothing, steel-toe boots, safety glasses.",
        "LOTO: notify, shut down, isolate, lock/tag, release energy, verify zero energy.",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
print("=== Running Ragas Evaluation ===")
print("(This may take a minute — it makes multiple LLM calls per sample)\n")

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm,
    embeddings=embeddings,
)

print("=== Results ===")
print(f"Faithfulness:      {results['faithfulness']:.3f}")
print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}")
print(f"Context Precision: {results['context_precision']:.3f}")

# Per-sample breakdown
print("\n=== Per-Sample Breakdown ===")
df = results.to_pandas()
for _, row in df.iterrows():
    print(f"\nQ: {row['question'][:60]}...")
    print(f"  Faithfulness:      {row['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {row['answer_relevancy']:.3f}")
    print(f"  Context Precision: {row['context_precision']:.3f}")

print("\n=== What These Scores Mean ===")
print("Faithfulness > 0.8:  Answers are well-grounded in context (low hallucination)")
print("Answer Relevancy > 0.8:  Answers address the actual question")
print("Context Precision > 0.8:  Retrieved docs are actually relevant")
print("\nScores < 0.7 indicate areas that need improvement.")
```

---

## Exercise 2: Evaluating Your RAG Pipeline End-to-End

```python
# 10-rag-evaluation-with-ragas/ex2_end_to_end_eval.py
"""Connect Ragas to your actual RAG pipeline and evaluate it."""

import chromadb
import ollama
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings

# === Step 1: Set up your RAG pipeline (from Module 06) ===
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="eval_test_kb")

# Your knowledge base
docs = [
    {"id": "MT-302", "text": "Torque Specification MT-302: Frame #4200. M8=25-30Nm. M10=45-55Nm. M12=80-100Nm. Star pattern. Calibrated wrench ±2%. QC sampling 10%. Form QC-110."},
    {"id": "WPS-201", "text": "Weld Procedure WPS-201: GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 shielding gas at 25-30 CFH. No preheat under 1 inch. Visual + UT per AWS D1.1."},
    {"id": "QC-107", "text": "Inspection Form QC-107: Visual/dimensional inspection. Check surface finish, weld quality, hardware, paint. All items must pass. Fail = red HOLD tag + notify supervisor."},
    {"id": "SAFE-001", "text": "LOTO Procedure: Notify operators. Normal shutdown. Isolate all energy (electrical, hydraulic, pneumatic). Personal lock and tag. Release stored energy. Verify zero energy. Only lock owner removes."},
    {"id": "PPE-001", "text": "PPE Requirements: Safety glasses always in production. Hearing protection above 85dB. Steel-toe boots required. Welding: auto-dark helmet shade 10-13, leather gloves, FR clothing."},
]

collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
)


def rag_query(question: str) -> dict:
    """Your RAG pipeline — returns answer + context."""
    results = collection.query(query_texts=[question], n_results=2)
    context_docs = results["documents"][0]

    context_str = "\n".join(context_docs)
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "Answer using ONLY the provided context. Cite sources."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )

    return {
        "answer": response["message"]["content"],
        "contexts": context_docs,
    }


# === Step 2: Generate test cases ===
test_questions = [
    {"q": "What torque do M10 bolts need on Frame #4200?", "gt": "M10 bolts on Frame #4200 need 45-55 Nm per MT-302."},
    {"q": "What shielding gas for MIG welding steel?", "gt": "75% Argon / 25% CO2 at 25-30 CFH per WPS-201."},
    {"q": "What happens when a part fails inspection?", "gt": "Apply red HOLD tag and notify shift supervisor per QC-107."},
    {"q": "Who can remove a lockout tag?", "gt": "Only the person who applied the lock can remove it per SAFE-001."},
    {"q": "When is hearing protection required?", "gt": "Hearing protection required above 85dB per PPE-001."},
]

# === Step 3: Run RAG and collect results ===
print("=== Running RAG Pipeline on Test Cases ===\n")

questions, answers, contexts, ground_truths = [], [], [], []

for test in test_questions:
    result = rag_query(test["q"])
    questions.append(test["q"])
    answers.append(result["answer"])
    contexts.append(result["contexts"])
    ground_truths.append(test["gt"])
    print(f"Q: {test['q']}")
    print(f"A: {result['answer'][:100]}...\n")

# === Step 4: Evaluate with Ragas ===
print("=== Running Ragas Evaluation ===\n")

llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0.0))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings,
)

print("=== OVERALL SCORES ===")
print(f"Faithfulness:      {results['faithfulness']:.3f}  (Are answers grounded in context?)")
print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}  (Do answers address the question?)")
print(f"Context Precision: {results['context_precision']:.3f}  (Are retrieved docs relevant?)")
print(f"Context Recall:    {results['context_recall']:.3f}  (Did we retrieve all needed info?)")

# Find weak spots
df = results.to_pandas()
print("\n=== WEAKEST RESULTS (investigate these) ===")
for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    worst_idx = df[metric].idxmin()
    worst_row = df.iloc[worst_idx]
    if worst_row[metric] < 0.8:
        print(f"\n⚠️  Low {metric} ({worst_row[metric]:.3f})")
        print(f"   Question: {worst_row['question']}")
```

---

## Exercise 3: Before/After Comparison

```python
# 10-rag-evaluation-with-ragas/ex3_before_after.py
"""Measure the impact of RAG improvements using Ragas scores."""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
import json

# Simulated before/after scenario:
# BEFORE: Basic RAG with poor chunking
# AFTER: Improved RAG with better chunking + system prompt

comparison = {
    "questions": [
        "What is the maximum interpass temperature for welding?",
        "What form do I use for visual inspection?",
        "What's the torque pattern for Frame #4200?",
    ],
    "before": {
        "answers": [
            "The welding procedure mentions temperature requirements.",
            "You should use an inspection form.",
            "Torque the bolts on Frame #4200.",
        ],
        "contexts": [
            ["WPS-201: GMAW welding of carbon steel. Various temperature and gas requirements apply."],
            ["Quality forms are used for different inspection types in the facility."],
            ["MT-302 covers torque for Frame #4200 assemblies."],
        ],
    },
    "after": {
        "answers": [
            "The maximum interpass temperature is 400°F per WPS-201.",
            "Use Form QC-107 for visual inspection. Record inspector badge number and date.",
            "Use a star pattern per the torque map diagram, specification MT-302.",
        ],
        "contexts": [
            ["WPS-201: Interpass Temperature: 400°F maximum. Preheat: Not required for material under 1 inch thick."],
            ["Form QC-107: Visual/dimensional inspection. Required fields: part number, lot number, inspector ID, date."],
            ["MT-302: Sequence: Star pattern per torque map diagram. Tool: Calibrated torque wrench ±2%."],
        ],
    },
    "ground_truths": [
        "Maximum interpass temperature is 400°F per WPS-201.",
        "Form QC-107 for visual inspection per quality procedures.",
        "Star pattern per the torque map diagram, specification MT-302.",
    ],
}

# Evaluate both
llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0.0))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

for version in ["before", "after"]:
    print(f"\n{'='*40}")
    print(f"  {version.upper()} IMPROVEMENT")
    print(f"{'='*40}")

    dataset = Dataset.from_dict({
        "question": comparison["questions"],
        "answer": comparison[version]["answers"],
        "contexts": comparison[version]["contexts"],
        "ground_truth": comparison["ground_truths"],
    })

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    print(f"Faithfulness:     {results['faithfulness']:.3f}")
    print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")

print("\n=== The Power of Before/After ===")
print("This is how you PROVE improvements to stakeholders:")
print("  'After improving our chunking strategy and adding specific context,")
print("   faithfulness improved from X to Y and relevancy from A to B.'")
print("\nThis is the data that gets projects approved and budgets allocated.")
```

---

## Takeaways

1. **Ragas is your go-to RAG evaluation framework** — research-backed metrics, easy to use
2. **Faithfulness catches hallucinations** — the most critical metric for manufacturing (safety!)
3. **Context Precision tells you if retrieval is working** — bad retrieval = bad answers
4. **Reference-free metrics** (faithfulness, relevancy) work on production data without ground-truth labels
5. **Before/after comparisons** are how you prove improvements to stakeholders

## Setting the Stage for Module 11

Ragas evaluates your RAG pipeline holistically. Module 11 introduces **DeepEval** — a pytest-compatible testing framework for LLMs. It lets you write LLM tests that run in CI/CD just like unit tests, catching regressions before they reach production.
