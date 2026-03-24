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
pip install ragas datasets langchain-openai --quiet
```

Now create a file:

```python
# 10-rag-evaluation-with-ragas/step1_first_metric.py

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
```

No errors? Good. Let's set up the local models Ragas will use as its judge:

```python
# Ragas needs an LLM (for judging) and embeddings (for similarity)
# We use Ollama's OpenAI-compatible endpoint so patterns transfer to production
judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="gemma3:12b",
        temperature=0.0,
    )
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="nomic-embed-text",
    )
)
```

### Step 2: One test case

Ragas evaluates individual samples. Each sample is a question + answer + context (and optionally a ground truth). Let's create exactly one:

```python
sample = SingleTurnSample(
    user_input="What is the quarterback's completion rate and release time?",
    response="The quarterback completes 68% of passes with a 2.3-second average release time, per scouting report QB-101.",
    retrieved_contexts=[
        "QB-101: Pocket passer with elite accuracy. Completes 68% of passes "
        "with 2.3-second average release. Excels on intermediate routes (15-25 yards). "
        "Reads defenses pre-snap. Arm strength: 62 mph."
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

You should see something like `0.95` or `1.0`. That means the answer is well-grounded in the context -- it didn't make anything up. Every claim in the answer ("68%", "2.3-second", "QB-101") can be traced back to the retrieved document.

### Step 4: Now break it on purpose

What happens when the LLM hallucinates? Let's create a bad example:

```python
bad_sample = SingleTurnSample(
    user_input="What is the quarterback's completion rate and release time?",
    response="The quarterback completes 82% of passes with a 1.9-second release. He is a dual-threat who rushed for 800 yards. Comparable to Patrick Mahomes.",
    retrieved_contexts=[
        "QB-101: Pocket passer with elite accuracy. Completes 68% of passes "
        "with 2.3-second average release. Excels on intermediate routes (15-25 yards). "
        "Reads defenses pre-snap. Arm strength: 62 mph."
    ],
)

bad_score = asyncio.run(faithfulness.single_turn_ascore(bad_sample))
print(f"Faithfulness (hallucinated): {bad_score:.2f}")
```

The score should drop significantly -- maybe `0.2` or `0.3`. Ragas caught it. The answer claims 82% completion (wrong), mentions rushing yards (not in context), and cites a Mahomes comparison (not in context). Three hallucinations in one answer, and faithfulness flagged them.

This is why faithfulness is the most critical metric for football scouting. Wrong stats can lead to bad draft picks. Fabricated measurables can cost a franchise millions.

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
    user_input="What is the quarterback's completion rate and release time?",
    response="The quarterback played at a Big Ten school. He started 3 seasons and was team captain. His coach praised his leadership in interviews.",
    retrieved_contexts=[
        "QB-101: Pocket passer with elite accuracy. Completes 68% of passes "
        "with 2.3-second average release. Arm strength: 62 mph."
    ],
)

score = asyncio.run(relevancy.single_turn_ascore(offtopic_sample))
print(f"Answer relevancy (off-topic): {score:.2f}")
```

The answer is factual-sounding, but it completely ignores the question. Relevancy catches this -- faithfulness alone might not, because the answer isn't necessarily *unfaithful* if that background info were in the context.

### Step 6: Context precision

This one evaluates your *retriever*, not the LLM. Did you pull back the right documents?

```python
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=judge_llm)

# Good retrieval -- the context is exactly what we need
good_retrieval = SingleTurnSample(
    user_input="Who has the best pass protection among the prospects?",
    response="The offensive lineman in report OL-401 has excellent pass protection, allowing only 2 sacks in 580 snaps with 34-inch arms.",
    retrieved_contexts=[
        "OL-401: Excellent pass protection anchor. Quick lateral movement. 34-inch arms. Run blocking: 82.5/100. 2 sacks in 580 snaps. Weakness: combo blocks.",
        "RB-201: Explosive runner with 4.38 40-yard dash. Exceptional vision. Weakness: pass protection and blitz pickup.",
    ],
)

score = asyncio.run(context_precision.single_turn_ascore(good_retrieval))
print(f"Context precision (good retrieval): {score:.2f}")
```

Now try with irrelevant context mixed in:

```python
noisy_retrieval = SingleTurnSample(
    user_input="Who has the best pass protection among the prospects?",
    response="The offensive lineman has good pass protection skills.",
    retrieved_contexts=[
        "The team cafeteria serves lunch from 11:30 to 1:00 daily.",
        "The practice facility turf was replaced last summer.",
        "OL-401: Excellent pass protection anchor. Quick lateral movement. 34-inch arms.",
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
    user_input="What is the complete scouting profile for the running back?",
    response="The running back has a fast 40-yard dash time.",
    retrieved_contexts=[
        "RB-201: Explosive runner with 4.38 40-yard dash."
    ],
    reference="RB-201: Explosive runner with 4.38 40-yard dash. Exceptional vision, finds cutback lanes. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection and blitz pickup.",
)

score = asyncio.run(context_recall.single_turn_ascore(incomplete_retrieval))
print(f"Context recall (incomplete): {score:.2f}")
```

Low score -- the retrieved context only covers the 40 time from a full scouting profile. The retriever missed most of the relevant information. In football scouting, this is costly: a GM seeing only speed data might overdraft a one-dimensional back who can't pass protect.

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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="gemma3:12b",
        temperature=0.0,
    )
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="nomic-embed-text",
    )
)

samples = [
    SingleTurnSample(
        user_input="What is the quarterback's completion rate?",
        response="The QB completes 68% of passes with a 2.3-second release per QB-101.",
        retrieved_contexts=[
            "QB-101: Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release."
        ],
    ),
    SingleTurnSample(
        user_input="Who has the best pass protection?",
        response="The OL prospect in OL-401 has excellent pass protection, allowing only 2 sacks in 580 snaps.",
        retrieved_contexts=[
            "OL-401: Excellent pass protection anchor. Quick lateral movement. "
            "34-inch arms. 2 sacks in 580 snaps."
        ],
    ),
    SingleTurnSample(
        user_input="What is the running back's yards after contact?",
        response="The RB averages 3.8 yards after contact per RB-201.",
        retrieved_contexts=[
            "RB-201: Explosive runner with 4.38 40-yard dash. 3.8 yards after contact. 45 receptions out of backfield."
        ],
    ),
    SingleTurnSample(
        user_input="What is the wide receiver's drop rate?",
        response="The WR has a 2.1% drop rate per WR-301.",
        retrieved_contexts=[
            "WR-301: Crisp route runner with elite separation. 4.42 speed, 38-inch vertical. 2.1% drop rate."
        ],
    ),
    SingleTurnSample(
        user_input="What is the defense's base coverage scheme?",
        response="The defense runs Cover-3 base with single-high safety and press corners per DEF-501.",
        retrieved_contexts=[
            "DEF-501: Cover-3 base with single-high safety. Press corners. Pattern-match zone on 3rd down."
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

This is how you find the questions your RAG pipeline struggles with. Maybe it retrieves the wrong scouting report for defensive scheme questions, or maybe the offensive line answers add details not in the context. Each low score points you to a specific fix.

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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="gemma3:12b",
        temperature=0.0,
    )
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="nomic-embed-text",
    )
)

faithfulness_metric = Faithfulness(llm=judge_llm)
relevancy_metric = ResponseRelevancy(
    llm=judge_llm, embeddings=judge_embeddings
)

# BEFORE: Basic RAG with vague chunking, generic prompt
before_samples = [
    SingleTurnSample(
        user_input="What is the quarterback's arm strength?",
        response="The quarterback has good arm strength.",
        retrieved_contexts=[
            "QB-101: Various physical and performance attributes "
            "were measured at the combine."
        ],
    ),
    SingleTurnSample(
        user_input="What are the running back's weaknesses?",
        response="The running back has some areas to improve.",
        retrieved_contexts=[
            "RB-201: Running back prospect with various strengths and weaknesses."
        ],
    ),
    SingleTurnSample(
        user_input="What is the defensive scheme's weakness?",
        response="The defense has some vulnerabilities.",
        retrieved_contexts=[
            "DEF-501: Defensive scheme with various coverage concepts."
        ],
    ),
]

# AFTER: Improved chunking + specific system prompt
after_samples = [
    SingleTurnSample(
        user_input="What is the quarterback's arm strength?",
        response="The quarterback's arm strength measures 62 mph per scouting report QB-101.",
        retrieved_contexts=[
            "QB-101: Arm strength: 62 mph. Completes 68% of passes "
            "with 2.3-second average release. Excels on intermediate routes."
        ],
    ),
    SingleTurnSample(
        user_input="What are the running back's weaknesses?",
        response="The running back's primary weaknesses are pass protection and blitz pickup per RB-201.",
        retrieved_contexts=[
            "RB-201: Weakness: pass protection and blitz pickup. "
            "Explosive runner with 4.38 40-yard dash. 3.8 yards after contact."
        ],
    ),
    SingleTurnSample(
        user_input="What is the defensive scheme's weakness?",
        response="The defense is vulnerable to crossing routes against zone coverage per DEF-501.",
        retrieved_contexts=[
            "DEF-501: Cover-3 base with single-high safety. Press corners. "
            "Weakness: crossing routes against zone."
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

Now you can walk into a meeting and say: "After improving our chunking strategy and adding position-specific context, faithfulness improved from 0.45 to 0.92 and relevancy from 0.50 to 0.88."

That is the data that gets projects approved and budgets allocated.

---

## Part 5: Connect to Your Real RAG Pipeline

### Step 12: Wire up your actual pipeline from Module 06

If you built the RAG pipeline in Module 06, let's evaluate it for real:

```python
# 10-rag-evaluation-with-ragas/step12_real_pipeline.py
import chromadb
from openai import OpenAI
import asyncio
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Your RAG pipeline ---
llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="eval_test_kb")

docs = [
    {"id": "QB-101", "text": "QB-101: Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Excels on intermediate routes (15-25 yards). Reads defenses pre-snap. Arm strength: 62 mph. Weakness: locks onto first read under pressure."},
    {"id": "RB-201", "text": "RB-201: Explosive runner with 4.38 40-yard dash. Exceptional vision, finds cutback lanes. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection and blitz pickup."},
    {"id": "WR-301", "text": "WR-301: Crisp route runner with elite separation. Full route tree, slot and outside. 4.42 speed, 38-inch vertical. 2.1% drop rate. Weakness: press coverage at the line."},
    {"id": "OL-401", "text": "OL-401: Excellent pass protection anchor. Quick lateral movement. 34-inch arms. Run blocking: 82.5/100. 2 sacks in 580 snaps. Weakness: combo blocks."},
    {"id": "DEF-501", "text": "DEF-501: Cover-3 base with single-high safety. Press corners. Pattern-match zone on 3rd down. Aggressive nickel blitz. Weakness: crossing routes against zone."},
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
    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {
                "role": "system",
                "content": "Answer using ONLY the provided context. Be specific. Cite scouting report IDs.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}",
            },
        ],
        temperature=0.0,
    )
    return {
        "answer": response.choices[0].message.content,
        "contexts": context_docs,
    }
```

### Step 13: Generate answers and evaluate

```python
test_questions = [
    "What is the quarterback's completion rate?",
    "Who has the best pass protection?",
    "What are the running back's weaknesses?",
    "What is the wide receiver's drop rate?",
    "What is the defensive scheme's base coverage?",
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
    ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="gemma3:12b",
        temperature=0.0,
    )
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="nomic-embed-text",
    )
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

1. **Faithfulness is your most critical metric** -- in football scouting, hallucinated stats or fabricated measurables can lead to costly draft mistakes.
2. **Context precision tells you if retrieval is working** -- bad retrieval means bad answers regardless of how good your LLM is.
3. **Reference-free metrics** (faithfulness, relevancy) work on production data without pre-labeled ground truth.
4. **Before/after comparisons** are how you prove improvements to stakeholders with hard numbers.
5. **Everything runs locally** with Ollama -- no API keys, no data leaving your machine.

## What's Next

Ragas evaluates your RAG pipeline holistically. Module 11 introduces **DeepEval** -- a pytest-compatible testing framework. It lets you write LLM tests that run in CI/CD just like unit tests, catching regressions before they hit production.
