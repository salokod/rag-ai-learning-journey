# Module 06: RAG Fundamentals

## The Problem We're Solving

Let's start by seeing a real problem. Open a Python shell:

```bash
python3
```

Now ask Ollama about YOUR scouting data:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "What's the arm strength for prospect QB-101?"}],
)
print(response.choices[0].message.content)
```

What did you get? Probably one of two things:
- "I don't have information about QB-101" (honest but useless)
- A made-up scouting report that sounds confident but is completely wrong (dangerous)

**That's the problem.** The LLM doesn't know YOUR scouting data. It was trained on the internet, not your team's scouting reports.

Let's fix that.

---

## Step 1: Load Some Scouting Reports into ChromaDB

First, let's set up a small knowledge base. Think of this as a film library the LLM can search through.

Create a new file:

```bash
touch 06-rag-fundamentals/rag_workshop.py
```

Start with your imports and some sample scouting reports:

```python
import chromadb
from openai import OpenAI

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

scouting_docs = [
    {
        "id": "QB-101",
        "text": "Pocket passer with elite accuracy. Completes 68% of passes "
        "with a 2.3-second average release time. Excels on intermediate "
        "routes (15-25 yards) with anticipation throws. Reads defenses "
        "pre-snap and adjusts protection. Arm strength measured at 62 mph "
        "at the combine. Weakness: locks onto first read under heavy pressure.",
        "metadata": {"position": "QB", "report_type": "scouting"},
    },
    {
        "id": "RB-201",
        "text": "Explosive runner with 4.38 40-yard dash. Exceptional vision "
        "through traffic and finds cutback lanes consistently. Averages "
        "3.8 yards after contact per carry. Reliable pass catcher out of "
        "the backfield with 45 receptions last season. Weakness: needs to "
        "improve pass protection and blitz pickup.",
        "metadata": {"position": "RB", "report_type": "scouting"},
    },
    {
        "id": "WR-301",
        "text": "Crisp route runner with elite separation at the top of routes. "
        "Runs the full route tree from slot and outside. 4.42 speed with a "
        "38-inch vertical leap. Reliable hands with a 2.1% drop rate. "
        "Weakness: struggles against physical press coverage at the line "
        "of scrimmage.",
        "metadata": {"position": "WR", "report_type": "scouting"},
    },
    {
        "id": "OL-401",
        "text": "Excellent anchor in pass protection with quick lateral movement "
        "to mirror speed rushers. 34-inch arm length provides leverage "
        "advantage. Run blocking grade: 82.5 out of 100. Allowed only 2 "
        "sacks in 580 pass-blocking snaps last season. Weakness: combo "
        "blocks to the second level.",
        "metadata": {"position": "OL", "report_type": "scouting"},
    },
    {
        "id": "DEF-501",
        "text": "Cover-3 base defense with single-high safety. Corners play "
        "press technique on early downs. Linebackers run pattern-match zone "
        "on 3rd and long. Aggressive blitz from nickel and dime personnel. "
        "Tendency: susceptible to crossing routes against zone coverage.",
        "metadata": {"position": "DEF", "report_type": "scheme"},
    },
    {
        "id": "SPEC-601",
        "text": "Punter averages 46.2 yards per punt with 4.1-second hang time. "
        "Directional kicking grade: elite. Coffin corner accuracy: 73% "
        "inside the 10-yard line. Kickoff specialist reaches the end zone "
        "on 88% of attempts. Coverage units allow 7.2 average return yards.",
        "metadata": {"position": "SPEC", "report_type": "scouting"},
    },
]
```

Now load them into ChromaDB:

```python
client = chromadb.Client()
collection = client.create_collection(name="scouting_kb")

collection.add(
    ids=[doc["id"] for doc in scouting_docs],
    documents=[doc["text"] for doc in scouting_docs],
    metadatas=[doc["metadata"] for doc in scouting_docs],
)
print(f"Loaded {collection.count()} documents")
```

Run it. You should see `Loaded 6 documents`. That's your knowledge base -- six scouting reports, embedded and searchable.

---

## Step 2: Retrieval -- Finding the Right Documents

Before we involve the LLM at all, let's just do the **retrieval** part. Ask ChromaDB to find documents relevant to our original question:

```python
results = collection.query(
    query_texts=["Who has the strongest arm in this draft class?"],
    n_results=2,
)

for doc_id, doc_text, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["distances"][0],
):
    print(f"\n--- {doc_id} (distance: {distance:.3f}) ---")
    print(doc_text[:200])
```

Run this. What do you see?

It should pull back `QB-101` (the quarterback scouting report with the 62 mph arm strength) as the top result. Notice the distance score -- lower means more relevant.

Let's try another query:

```python
results = collection.query(
    query_texts=["Which prospect catches the ball best?"],
    n_results=2,
)

for doc_id in results["ids"][0]:
    print(doc_id)
```

Did it find `WR-301`? That document mentions reliable hands and a 2.1% drop rate. ChromaDB understood that "catches the ball best" is semantically related to drop rates and reliable hands even though the doc never uses the phrase "catches the ball."

That's the **Retrieve** step working. Now let's give these results to the LLM.

---

## Step 3: Augmentation -- Building the Prompt

Here's the key idea: we take the retrieved documents and stuff them into the prompt. The LLM reads them as context, then answers based on what it read.

Let's build an augmented prompt:

```python
question = "What's the arm strength for the quarterback prospect?"

results = collection.query(query_texts=[question], n_results=2)

context = "\n\n---\n\n".join(
    f"[Source: {doc_id}]\n{doc}"
    for doc_id, doc in zip(results["ids"][0], results["documents"][0])
)

augmented_prompt = f"""Answer the question based ONLY on the provided context documents.
If the answer is not in the context, say "I don't have that information in my documents."
Always cite which source document(s) you used.

CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""

print(augmented_prompt)
```

Run it and look at what gets printed. Notice how the prompt now contains the actual scouting report. The LLM won't have to guess -- the answer is right there in front of it.

---

## Step 4: Generation -- The LLM Answers with Real Data

Now send that augmented prompt to the LLM:

```python
response = llm.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": "You are an NFL draft analyst. "
            "Answer questions using only the provided scouting reports. "
            "Be specific and cite sources.",
        },
        {"role": "user", "content": augmented_prompt},
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
```

What did you get?

It should say something like: "Per scouting report QB-101, the quarterback prospect has an arm strength of 62 mph, measured at the combine."

**The LLM just cited a REAL scouting report from YOUR documents.** That's RAG.

---

## Step 5: Side-by-Side -- Without RAG vs. With RAG

Let's see the difference clearly. Add this to your script:

```python
question = "Which running back has the best vision and explosiveness?"

# WITHOUT RAG
print("=== WITHOUT RAG ===")
no_rag = llm.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": question}],
    temperature=0.0,
)
print(no_rag.choices[0].message.content[:300])

# WITH RAG
print("\n=== WITH RAG ===")
results = collection.query(query_texts=[question], n_results=2)
context = "\n\n".join(
    f"[{doc_id}]: {doc}"
    for doc_id, doc in zip(results["ids"][0], results["documents"][0])
)

with_rag = llm.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": "Answer using only the provided documents. Cite sources.",
        },
        {
            "role": "user",
            "content": f"Documents:\n{context}\n\nQuestion: {question}",
        },
    ],
    temperature=0.0,
)
print(with_rag.choices[0].message.content[:300])
```

Run it. Notice the difference:
- Without RAG: generic answer about NFL running backs from its training data, not YOUR prospects
- With RAG: cites RB-201, mentions the specific 4.38 40-yard dash, exceptional vision through traffic, and 3.8 yards after contact

In football scouting, the difference between "generally correct" and "per YOUR scouting reports" is everything.

---

## Step 6: What Happens When the Answer ISN'T in the Docs?

This is just as important. Try asking something your knowledge base doesn't cover:

```python
question = "What's this prospect's injury history?"

results = collection.query(query_texts=[question], n_results=2)
context = "\n\n".join(results["documents"][0])

response = llm.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": "Answer using only the provided documents. "
            "If the answer is not in the documents, say "
            "'I don't have that information in my documents.'",
        },
        {
            "role": "user",
            "content": f"Documents:\n{context}\n\nQuestion: {question}",
        },
    ],
    temperature=0.0,
)
print(response.choices[0].message.content)
```

It should say something like "I don't have that information in my documents."

**That's a GOOD answer.** In football scouting, "I don't know" is infinitely better than a hallucinated injury report that could tank a draft pick. The system prompt instruction to say "I don't have that information" is your safety net.

---

## Step 7: Build the RAG Function

Now let's wrap what we've built into a clean, reusable function. Each piece should look familiar -- we're just combining the steps:

```python
def rag_query(question: str, n_results: int = 2) -> dict:
    """Complete RAG pipeline: retrieve, augment, generate."""

    # RETRIEVE
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]
    distances = results["distances"][0]

    # AUGMENT
    context = "\n\n---\n\n".join(
        f"[Source: {doc_id}]\n{doc}"
        for doc_id, doc in zip(retrieved_ids, retrieved_docs)
    )

    augmented_prompt = f"""Answer based ONLY on the provided context.
If the answer isn't in the context, say "I don't have that information."
Cite your source document(s).

CONTEXT:
{context}

QUESTION: {question}"""

    # GENERATE
    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {
                "role": "system",
                "content": "You are an NFL draft analyst. "
                "Be specific and cite sources.",
            },
            {"role": "user", "content": augmented_prompt},
        ],
        temperature=0.0,
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": retrieved_ids,
        "distances": distances,
    }
```

Try it with a few questions:

```python
result = rag_query("Who has the strongest arm in this draft class?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer'][:300]}")
```

```python
result = rag_query("What receiver runs the best routes?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer'][:300]}")
```

```python
result = rag_query("What are the defensive tendencies on third down?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer'][:300]}")
```

Notice how the last one pulls from the DEF-501 scheme report -- "Linebackers run pattern-match zone on 3rd and long." That's a real scouting report from your docs, not a generic answer.

---

## Step 8: Metadata Filtering -- Narrowing the Search

When you have hundreds of scouting reports, you don't always want to search everything. Imagine a scout focused on skill position players -- they probably only need offensive skill reports.

Let's try filtering by metadata:

```python
# Search ONLY for scouting reports (not scheme reports)
results = collection.query(
    query_texts=["speed and athleticism"],
    n_results=3,
    where={"report_type": "scouting"},
)
for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
    print(f"[{doc_id}] {doc[:80]}...")
```

Without the filter, "speed and athleticism" might match the defensive scheme report DEF-501 too. With the scouting report filter, it only returns individual player evaluations. That's more precise when you know what type of report you're looking for.

Try combining filters:

```python
# Search for only QB scouting reports
results = collection.query(
    query_texts=["passing accuracy"],
    n_results=3,
    where={"position": "QB"},
)
for doc_id in results["ids"][0]:
    print(doc_id)
```

This returns only quarterback reports, filtering out other positions. Think of it like asking the film library "show me only the quarterback tape, not the wide receiver reels."

---

## Step 9: RAG-Powered Scouting Report Generator

Now let's put it all together for something genuinely useful -- generating scouting reports grounded in your team's actual evaluation data.

First, add some reference documents about your scouting process and grading standards:

```python
# Create a fresh collection with draft-focused docs
report_collection = client.create_collection(name="report_kb")

report_docs = [
    {
        "id": "ref-combine",
        "text": "Combine benchmarks by position: QB arm strength 55+ mph elite, "
        "RB 40-yard dash sub-4.45 elite, WR vertical leap 36+ inches elite, "
        "OL 5.0 or faster 40-yard dash. All measurables logged per prospect ID.",
    },
    {
        "id": "ref-grading",
        "text": "Grading scale: 9.0+ generational talent, 7.5-8.9 Pro Bowl caliber, "
        "6.0-7.4 solid starter, 5.0-5.9 backup/roster player, below 5.0 undraftable. "
        "Grade based on tape, production, athleticism, and character.",
    },
    {
        "id": "ref-film-review",
        "text": "Film review process: minimum 3 full game cutups per prospect. "
        "Grade each play on effort, technique, and football IQ. "
        "Note tendencies, alignments, and snap-to-snap consistency.",
    },
    {
        "id": "ref-draft-board",
        "text": "Draft board format: Rank by overall grade, then by position tier. "
        "Include prospect name, school, height/weight, grade, and one-line summary. "
        "Flag medical concerns and character notes separately.",
    },
    {
        "id": "ref-format",
        "text": "Scouting report format: Title in ALL CAPS, 3-5 key traits starting "
        "with action verbs, weakness section at bottom, reference to applicable "
        "combine data and film grades.",
    },
    {
        "id": "ref-scheme-fit",
        "text": "Scheme fit evaluation: grade prospect fit for West Coast, spread, "
        "air raid, and power run schemes. Note which offensive or defensive system "
        "maximizes the player's strengths and minimizes weaknesses.",
    },
]

report_collection.add(
    ids=[d["id"] for d in report_docs],
    documents=[d["text"] for d in report_docs],
)
print(f"Loaded {report_collection.count()} reference docs")
```

Now build the report generator:

```python
def generate_report(report_request: str) -> str:
    """Generate a scouting report grounded in team docs."""

    results = report_collection.query(query_texts=[report_request], n_results=3)
    context = "\n".join(
        f"- [{doc_id}]: {doc}"
        for doc_id, doc in zip(results["ids"][0], results["documents"][0])
    )

    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {
                "role": "system",
                "content": "You are an NFL scouting report writer. "
                "Generate scouting reports using the team's reference documents. "
                "Follow the team's format exactly. "
                "Include specific grades, combine benchmarks, and film notes "
                "from the reference docs. "
                "Do NOT make up grades or stats -- use ONLY what's provided.",
            },
            {
                "role": "user",
                "content": f'Generate a scouting report for: "{report_request}"\n\n'
                f"Team reference documents:\n{context}",
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content
```

Try it:

```python
print(generate_report("Evaluate a quarterback with elite arm talent for our draft board"))
```

Look at the output. Does it reference the grading scale? The combine benchmarks for QB arm strength (55+ mph elite)? The scouting report format?

**Those are REAL grading criteria and benchmarks from YOUR documents.** The LLM didn't make them up -- it read them from the context we provided.

Try a couple more:

```python
print("=" * 60)
print(generate_report("Grade a running back prospect for scheme fit"))
```

```python
print("=" * 60)
print(generate_report("Build a film review summary for a wide receiver"))
```

Notice how each scouting report pulls different reference docs. The QB evaluation gets combine benchmarks and grading scale. The RB report gets scheme fit criteria. The WR film review gets the film review process docs. RAG finds the right context for each report.

---

## What You Built

Let's recap what just happened:

1. **Retrieve** -- ChromaDB finds documents relevant to the question using semantic search
2. **Augment** -- Those documents get stuffed into the prompt as context
3. **Generate** -- The LLM answers based on what it read, citing real sources

```
User Question --> ChromaDB Search --> Build Prompt with Docs --> LLM Answer
                  (Retrieve)         (Augment)                  (Generate)
```

That's the entire RAG pattern. Everything from here builds on this pipeline.

## Key Takeaways

- **RAG grounds the LLM in your data** -- it answers from your docs, not its training data
- **"I don't have that information" is a GOOD answer** -- way better than hallucinating
- **Metadata filtering** makes retrieval precise when you have many documents
- **The quality of retrieval determines the quality of answers** -- if you retrieve the wrong docs, you get wrong answers

That last point is the segue into Module 07. Our basic retrieval works, but it's not perfect. What happens when the search pulls back the wrong scouting reports? Let's find out.
