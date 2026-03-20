# Module 10: RAG Evaluation with Ragas

## Goal
Use Ragas -- the industry-standard RAG evaluation framework -- to measure whether your RAG pipeline retrieves the right context and generates faithful answers. Research-backed metrics, a few lines of code.

---

## Why Ragas?

In Module 09 you built evaluation from scratch. That works, but Ragas gives you something better: **metrics designed specifically for RAG**, backed by academic research. Instead of inventing your own scoring criteria, you get standardized measurements that the ML community agrees on.

The big ones:
- **Faithfulness** -- Is the answer grounded in the retrieved context? (catches hallucinations)
- **Answer Relevancy** -- Does the answer actually address the question?
- **Context Precision** -- Did the retriever return useful documents?
- **Context Recall** -- Did the retriever find ALL the needed information?

Let's try them one at a time.

---

## Part 1: Your First Ragas Metric

### Step 1: Install and import

Ragas should already be installed from your requirements.txt. Let's verify:

```bash
pip install ragas datasets langchain-ollama --quiet
```

Now create a file:

```python
# 10-rag-evaluation-with-ragas/step1_first_metric.py

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
```

No errors? Good. Let's set up the local models Ragas will use as its judge:

```python
# Ragas needs an LLM (for judging) and embeddings (for similarity)
# We use Ollama so everything stays local -- no API keys needed
judge_llm = LangchainLLMWrapper(
    ChatOllama(model="llama3.1:8b", temperature=0.0)
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)
```

### Step 2: One test case

Ragas evaluates individual samples. Each sample is a question + answer + context (and optionally a ground truth). Let's create exactly one:

```python
sample = SingleTurnSample(
    user_input="What is the torque spec for M10 bolts on Frame Assembly #4200?",
    response="The torque specification for M10 bolts on Frame Assembly #4200 is 45-55 Nm, per specification MT-302.",
    retrieved_contexts=[
        "Torque Specification MT-302 for Frame Assembly #4200: "
        "M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. "
        "Use calibrated torque wrench +/-2%."
    ],
)
```

Think about what we have here: a question, the RAG pipeline's answer, and the context that was retrieved. Now let's see if the answer is faithful to that context.

### Step 3: Run faithfulness

```python
import asyncio

faithfulness = Faithfulness(llm=judge_llm)

score = asyncio.run(faithfulness.single_turn_ascore(sample))
print(f"Faithfulness: {score:.2f}")
```

You should see something like `0.95` or `1.0`. That means the answer is well-grounded in the context -- it didn't make anything up. Every claim in the answer ("45-55 Nm", "MT-302") can be traced back to the retrieved document.

### Step 4: Now break it on purpose

What happens when the LLM hallucinates? Let's create a bad example:

```python
bad_sample = SingleTurnSample(
    user_input="What is the torque spec for M10 bolts on Frame Assembly #4200?",
    response="The torque specification for M10 bolts is 120 Nm. Always use an impact wrench for faster assembly. Refer to ISO 9001 for quality standards.",
    retrieved_contexts=[
        "Torque Specification MT-302 for Frame Assembly #4200: "
        "M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. "
        "Use calibrated torque wrench +/-2%."
    ],
)

bad_score = asyncio.run(faithfulness.single_turn_ascore(bad_sample))
print(f"Faithfulness (hallucinated): {bad_score:.2f}")
```

The score should drop significantly -- maybe `0.2` or `0.3`. Ragas caught it. The answer claims 120 Nm (wrong), mentions impact wrench (not in context), and cites ISO 9001 (not in context). Three hallucinations in one answer, and faithfulness flagged them.

This is why faithfulness is the most critical metric for manufacturing. Wrong torque specs can break parts. Wrong tool recommendations can injure people.

---

## Part 2: More Metrics, One at a Time

### Step 5: Answer relevancy

Faithfulness checks "is it grounded in context?" Answer relevancy checks "does it actually answer the question?"

```python
# 10-rag-evaluation-with-ragas/step5_relevancy.py
from ragas.metrics import ResponseRelevancy

relevancy = ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings)

# Good answer -- directly addresses the question
score = asyncio.run(relevancy.single_turn_ascore(sample))
print(f"Answer relevancy (good): {score:.2f}")
```

Now try an off-topic answer:

```python
offtopic_sample = SingleTurnSample(
    user_input="What is the torque spec for M10 bolts on Frame Assembly #4200?",
    response="Frame Assembly #4200 is manufactured at our Toledo plant. The assembly line runs two shifts, Monday through Friday. Quality inspection occurs at station 7.",
    retrieved_contexts=[
        "Torque Specification MT-302 for Frame Assembly #4200: "
        "M8=25-30Nm, M10=45-55Nm, M12=80-100Nm."
    ],
)

score = asyncio.run(relevancy.single_turn_ascore(offtopic_sample))
print(f"Answer relevancy (off-topic): {score:.2f}")
```

The answer is factual-sounding, but it completely ignores the question. Relevancy catches this -- faithfulness alone might not, because the answer isn't necessarily *unfaithful* if that plant info were in the context.

### Step 6: Context precision

This one evaluates your *retriever*, not the LLM. Did you pull back the right documents?

```python
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=judge_llm)

# Good retrieval -- the context is exactly what we need
good_retrieval = SingleTurnSample(
    user_input="What PPE is required for MIG welding?",
    response="MIG welding requires an auto-darkening helmet (shade 10-13), leather welding gloves, FR clothing, and steel-toe boots.",
    retrieved_contexts=[
        "PPE for welding: Auto-darkening helmet shade 10-13, leather welding gloves, FR clothing, steel-toe boots. Safety glasses under helmet.",
        "GMAW welding parameters: ER70S-6 wire, 75/25 Ar/CO2 at 25-30 CFH.",
    ],
)

score = asyncio.run(context_precision.single_turn_ascore(good_retrieval))
print(f"Context precision (good retrieval): {score:.2f}")
```

Now try with irrelevant context mixed in:

```python
noisy_retrieval = SingleTurnSample(
    user_input="What PPE is required for MIG welding?",
    response="Welding requires appropriate protective equipment.",
    retrieved_contexts=[
        "The cafeteria menu is updated weekly. Tuesday is taco day.",
        "Parking lot B will be resurfaced next month.",
        "PPE for welding: Auto-darkening helmet shade 10-13, leather gloves.",
    ],
)

score = asyncio.run(context_precision.single_turn_ascore(noisy_retrieval))
print(f"Context precision (noisy retrieval): {score:.2f}")
```

Lower score. Two of the three retrieved documents were completely useless. This tells you your retriever needs tuning -- maybe better embeddings, better chunking, or more documents in your knowledge base.

### Step 7: Context recall

This requires a ground truth answer -- it checks whether the retrieved context contains everything needed to produce the correct answer:

```python
from ragas.metrics import LLMContextRecall

context_recall = LLMContextRecall(llm=judge_llm)

# Missing context -- ground truth mentions things not in retrieved docs
incomplete_retrieval = SingleTurnSample(
    user_input="What is the complete LOTO procedure?",
    response="Lock out the machine and apply your tag.",
    retrieved_contexts=[
        "Apply personal lock and tag to energy isolation device."
    ],
    reference="LOTO: 1) Notify affected operators, 2) Normal shutdown, 3) Isolate all energy sources, 4) Apply personal lock and tag, 5) Release stored energy, 6) Verify zero energy state.",
)

score = asyncio.run(context_recall.single_turn_ascore(incomplete_retrieval))
print(f"Context recall (incomplete): {score:.2f}")
```

Low score -- the retrieved context only covers step 4 of a 6-step procedure. The retriever missed most of the relevant information. In manufacturing, this is dangerous: an operator following only step 4 skips the critical "verify zero energy" step.

---

## Part 3: Evaluating Multiple Samples at Once

### Step 8: Build a dataset

In real use, you evaluate many questions at once. Let's set up a batch:

```python
# 10-rag-evaluation-with-ragas/step8_batch_eval.py
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

judge_llm = LangchainLLMWrapper(
    ChatOllama(model="llama3.1:8b", temperature=0.0)
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

samples = [
    SingleTurnSample(
        user_input="What torque do M10 bolts need on Frame #4200?",
        response="M10 bolts on Frame #4200 need 45-55 Nm per MT-302.",
        retrieved_contexts=[
            "MT-302: Frame #4200. M8=25-30Nm. M10=45-55Nm. M12=80-100Nm."
        ],
    ),
    SingleTurnSample(
        user_input="What shielding gas for MIG welding steel?",
        response="Use 75% Argon / 25% CO2 at 25-30 CFH per WPS-201.",
        retrieved_contexts=[
            "WPS-201: GMAW carbon steel. ER70S-6 wire. "
            "75/25 Ar/CO2 shielding gas at 25-30 CFH."
        ],
    ),
    SingleTurnSample(
        user_input="What happens when a part fails inspection?",
        response="Apply red HOLD tag and notify shift supervisor per QC-107.",
        retrieved_contexts=[
            "QC-107: All items must pass. Fail = red HOLD tag + notify supervisor."
        ],
    ),
    SingleTurnSample(
        user_input="Who can remove a lockout tag?",
        response="Only the person who applied the lock can remove it.",
        retrieved_contexts=[
            "LOTO: Personal lock and tag. Only lock owner removes."
        ],
    ),
    SingleTurnSample(
        user_input="When is hearing protection required?",
        response="Hearing protection is required above 85dB per PPE-001.",
        retrieved_contexts=[
            "PPE-001: Hearing protection above 85dB. Steel-toe boots required."
        ],
    ),
]

dataset = EvaluationDataset(samples=samples)
```

### Step 9: Run batch evaluation

```python
print("Running Ragas evaluation on 5 samples...")
print("(This makes multiple LLM calls per sample -- give it a minute)\n")

faithfulness_metric = Faithfulness(llm=judge_llm)
relevancy_metric = ResponseRelevancy(
    llm=judge_llm, embeddings=judge_embeddings
)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness_metric, relevancy_metric],
)

print("=== Overall Scores ===")
print(f"Faithfulness:     {results['faithfulness']:.3f}")
print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
```

Now look at per-sample results to find weak spots:

```python
df = results.to_pandas()
print("\n=== Per-Sample Breakdown ===")
for _, row in df.iterrows():
    print(f"\nQ: {row['user_input'][:60]}...")
    print(f"  Faithfulness:     {row['faithfulness']:.2f}")
    print(f"  Answer Relevancy: {row['answer_relevancy']:.2f}")
    flag = ""
    if row['faithfulness'] < 0.8:
        flag += " [LOW FAITHFULNESS]"
    if row['answer_relevancy'] < 0.8:
        flag += " [LOW RELEVANCY]"
    if flag:
        print(f"  *** INVESTIGATE:{flag}")
```

This is how you find the questions your RAG pipeline struggles with. Maybe it retrieves the wrong document for LOTO questions, or maybe the welding answers add details not in the context. Each low score points you to a specific fix.

---

## Part 4: Before/After Comparison

This is the money shot -- proving that a change to your pipeline actually improved things.

### Step 10: Simulate basic vs. improved RAG

```python
# 10-rag-evaluation-with-ragas/step10_before_after.py
import asyncio
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

judge_llm = LangchainLLMWrapper(
    ChatOllama(model="llama3.1:8b", temperature=0.0)
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

faithfulness_metric = Faithfulness(llm=judge_llm)
relevancy_metric = ResponseRelevancy(
    llm=judge_llm, embeddings=judge_embeddings
)

# BEFORE: Basic RAG with vague chunking, generic prompt
before_samples = [
    SingleTurnSample(
        user_input="What is the maximum interpass temperature for welding?",
        response="The welding procedure mentions temperature requirements.",
        retrieved_contexts=[
            "WPS-201: GMAW welding of carbon steel. "
            "Various temperature and gas requirements apply."
        ],
    ),
    SingleTurnSample(
        user_input="What form do I use for visual inspection?",
        response="You should use an inspection form.",
        retrieved_contexts=[
            "Quality forms are used for different inspection types."
        ],
    ),
    SingleTurnSample(
        user_input="What's the torque pattern for Frame #4200?",
        response="Torque the bolts on Frame #4200.",
        retrieved_contexts=[
            "MT-302 covers torque for Frame #4200 assemblies."
        ],
    ),
]

# AFTER: Improved chunking + specific system prompt
after_samples = [
    SingleTurnSample(
        user_input="What is the maximum interpass temperature for welding?",
        response="The maximum interpass temperature is 400 degrees F per WPS-201.",
        retrieved_contexts=[
            "WPS-201: Interpass Temperature: 400F maximum. "
            "Preheat: Not required for material under 1 inch thick."
        ],
    ),
    SingleTurnSample(
        user_input="What form do I use for visual inspection?",
        response="Use Form QC-107 for visual inspection. Record inspector badge number and date.",
        retrieved_contexts=[
            "Form QC-107: Visual/dimensional inspection. "
            "Required fields: part number, lot number, inspector ID, date."
        ],
    ),
    SingleTurnSample(
        user_input="What's the torque pattern for Frame #4200?",
        response="Use a star pattern per the torque map diagram, specification MT-302.",
        retrieved_contexts=[
            "MT-302: Sequence: Star pattern per torque map diagram. "
            "Tool: Calibrated torque wrench +/-2%."
        ],
    ),
]
```

### Step 11: Score both and compare

```python
async def score_samples(samples):
    f_scores = []
    r_scores = []
    for s in samples:
        f = await faithfulness_metric.single_turn_ascore(s)
        r = await relevancy_metric.single_turn_ascore(s)
        f_scores.append(f)
        r_scores.append(r)
    return {
        "faithfulness": sum(f_scores) / len(f_scores),
        "relevancy": sum(r_scores) / len(r_scores),
    }

before_scores = asyncio.run(score_samples(before_samples))
after_scores = asyncio.run(score_samples(after_samples))

print("=== BEFORE (basic RAG) ===")
print(f"  Faithfulness:     {before_scores['faithfulness']:.2f}")
print(f"  Answer Relevancy: {before_scores['relevancy']:.2f}")

print("\n=== AFTER (improved RAG) ===")
print(f"  Faithfulness:     {after_scores['faithfulness']:.2f}")
print(f"  Answer Relevancy: {after_scores['relevancy']:.2f}")

print("\n=== IMPROVEMENT ===")
f_delta = after_scores['faithfulness'] - before_scores['faithfulness']
r_delta = after_scores['relevancy'] - before_scores['relevancy']
print(f"  Faithfulness:     {f_delta:+.2f}")
print(f"  Answer Relevancy: {r_delta:+.2f}")
```

Now you can walk into a meeting and say: "After improving our chunking strategy and adding document-specific context, faithfulness improved from 0.45 to 0.92 and relevancy from 0.50 to 0.88."

That is the data that gets projects approved and budgets allocated.

---

## Part 5: Connect to Your Real RAG Pipeline

### Step 12: Wire up your actual pipeline from Module 06

If you built the RAG pipeline in Module 06, let's evaluate it for real:

```python
# 10-rag-evaluation-with-ragas/step12_real_pipeline.py
import chromadb
import ollama
import asyncio
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

# --- Your RAG pipeline ---
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="eval_test_kb")

docs = [
    {"id": "MT-302", "text": "Torque Specification MT-302: Frame #4200. M8=25-30Nm. M10=45-55Nm. M12=80-100Nm. Star pattern. Calibrated wrench +/-2%."},
    {"id": "WPS-201", "text": "Weld Procedure WPS-201: GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH. Interpass temp 400F max. Visual + UT per AWS D1.1."},
    {"id": "QC-107", "text": "Inspection Form QC-107: Visual and dimensional inspection. Check surface finish, weld quality, hardware, paint. Fail = red HOLD tag + notify supervisor."},
    {"id": "SAFE-001", "text": "LOTO Procedure: Notify operators. Normal shutdown. Isolate all energy. Personal lock and tag. Release stored energy. Verify zero energy. Only lock owner removes."},
    {"id": "PPE-001", "text": "PPE Requirements: Safety glasses always in production. Hearing protection above 85dB. Steel-toe boots required. Welding: auto-dark helmet shade 10-13, leather gloves, FR clothing."},
]

collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
)


def rag_query(question):
    """Your RAG pipeline -- returns answer + contexts."""
    results = collection.query(query_texts=[question], n_results=2)
    context_docs = results["documents"][0]

    context_str = "\n".join(context_docs)
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "Answer using ONLY the provided context. Be specific. Cite document IDs.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}",
            },
        ],
        options={"temperature": 0.0},
    )
    return {
        "answer": response["message"]["content"],
        "contexts": context_docs,
    }
```

### Step 13: Generate answers and evaluate

```python
test_questions = [
    "What torque do M10 bolts need on Frame #4200?",
    "What shielding gas for MIG welding steel?",
    "What happens when a part fails inspection?",
    "Who can remove a lockout tag?",
    "When is hearing protection required?",
]

print("=== Querying RAG Pipeline ===\n")
samples = []
for q in test_questions:
    result = rag_query(q)
    print(f"Q: {q}")
    print(f"A: {result['answer'][:100]}...\n")
    samples.append(
        SingleTurnSample(
            user_input=q,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
        )
    )

# --- Evaluate ---
judge_llm = LangchainLLMWrapper(
    ChatOllama(model="llama3.1:8b", temperature=0.0)
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

dataset = EvaluationDataset(samples=samples)

print("=== Running Ragas Evaluation ===\n")

results = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        LLMContextPrecisionWithoutReference(llm=judge_llm),
    ],
)

print("=== YOUR RAG PIPELINE SCORES ===")
print(f"Faithfulness:      {results['faithfulness']:.3f}  (grounded in context?)")
print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}  (addresses the question?)")
print(f"Context Precision: {results['context_precision_without_reference']:.3f}  (retriever quality?)")

# Flag weak spots
df = results.to_pandas()
print("\n=== Questions Needing Attention ===")
for _, row in df.iterrows():
    issues = []
    if row.get('faithfulness', 1) < 0.8:
        issues.append(f"faithfulness={row['faithfulness']:.2f}")
    if row.get('answer_relevancy', 1) < 0.8:
        issues.append(f"relevancy={row['answer_relevancy']:.2f}")
    if issues:
        print(f"  {row['user_input']}")
        print(f"    Issues: {', '.join(issues)}")

if not any(
    row.get('faithfulness', 1) < 0.8 or row.get('answer_relevancy', 1) < 0.8
    for _, row in df.iterrows()
):
    print("  All questions passed! Your pipeline is looking solid.")
```

---

## A Note on the Ragas API

Ragas is actively evolving. The approach shown here uses `SingleTurnSample` and metric-specific constructors (e.g., `Faithfulness(llm=...)`), which is the current stable pattern. If you see older tutorials using `from ragas.metrics import faithfulness` (lowercase) with a global `evaluate()` that takes `llm=` and `embeddings=` directly, that was the previous API. Both may work depending on your installed version, but the approach in this module is the direction Ragas is heading.

If something breaks after a Ragas update, check their docs at https://docs.ragas.io for the latest API.

---

## Takeaways

1. **Faithfulness is your most critical metric** -- in manufacturing, hallucinated specs or procedures can cause real harm.
2. **Context precision tells you if retrieval is working** -- bad retrieval means bad answers regardless of how good your LLM is.
3. **Reference-free metrics** (faithfulness, relevancy) work on production data without pre-labeled ground truth.
4. **Before/after comparisons** are how you prove improvements to stakeholders with hard numbers.
5. **Everything runs locally** with Ollama -- no API keys, no data leaving your machine.

## What's Next

Ragas evaluates your RAG pipeline holistically. Module 11 introduces **DeepEval** -- a pytest-compatible testing framework. It lets you write LLM tests that run in CI/CD just like unit tests, catching regressions before they hit production.
