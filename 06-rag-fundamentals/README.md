# Module 06: RAG Fundamentals

## The Problem We're Solving

Let's start by seeing a real problem. Open a Python shell:

```bash
python3
```

Now ask Ollama about YOUR company's data:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "What's the torque spec for Assembly #4200?"}],
)
print(response.choices[0].message.content)
```

What did you get? Probably one of two things:
- "I don't have information about Assembly #4200" (honest but useless)
- A made-up torque spec that sounds confident but is completely wrong (dangerous)

**That's the problem.** The LLM doesn't know YOUR company's data. It was trained on the internet, not your SOPs.

Let's fix that.

---

## Step 1: Load Some Company Documents into ChromaDB

First, let's set up a small knowledge base. Think of this as a filing cabinet the LLM can search through.

Create a new file:

```bash
touch 06-rag-fundamentals/rag_workshop.py
```

Start with your imports and some sample manufacturing docs:

```python
import chromadb
from openai import OpenAI

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

manufacturing_docs = [
    {
        "id": "SOP-MT-302",
        "text": """TORQUE SPECIFICATION MT-302
Assembly: Frame Assembly #4200
All fasteners: Grade 8, zinc plated
M8 bolts: 25-30 Nm
M10 bolts: 45-55 Nm
M12 bolts: 80-100 Nm
Sequence: Star pattern per torque diagram
Tool: Calibrated torque wrench, +/-2% accuracy
Verification: 10% sampling by QC after assembly
Documentation: Record on Form QC-110""",
        "metadata": {"type": "specification", "department": "assembly"},
    },
    {
        "id": "WPS-201",
        "text": """WELDING PROCEDURE SPECIFICATION WPS-201
Process: GMAW (MIG Welding)
Base Metal: Carbon Steel (A36)
Filler Metal: ER70S-6, 0.035" diameter
Shielding Gas: 75% Argon / 25% CO2 at 25-30 CFH
Preheat: Not required for material under 1" thick
Interpass Temperature: 400F maximum
Post-Weld: Visual inspection required. UT for critical joints.
Acceptance Criteria: Per AWS D1.1 Section 6""",
        "metadata": {"type": "specification", "department": "welding"},
    },
    {
        "id": "SOP-QC-107",
        "text": """QUALITY CONTROL INSPECTION FORM QC-107
Purpose: Document visual and dimensional inspection results
Required fields: Part number, lot number, inspector ID, date
Visual inspection checklist:
- Surface finish (no scratches, dents, or tool marks beyond spec)
- Weld quality (no cracks, porosity, undercut, or incomplete fusion)
- Hardware installation (all fasteners present and torqued)
- Paint/coating (uniform coverage, no runs, drips, or bare spots)
Pass criteria: All checklist items must pass
Failure action: Tag with red HOLD tag, notify shift supervisor""",
        "metadata": {"type": "form", "department": "quality"},
    },
    {
        "id": "SOP-SAFE-001",
        "text": """LOCKOUT/TAGOUT PROCEDURE - ALL MACHINERY
Before maintenance or adjustment:
1. Notify affected operators
2. Shut down the machine using normal stop procedure
3. Isolate all energy sources (electrical, hydraulic, pneumatic)
4. Apply personal lock and tag to each energy isolation point
5. Release any stored energy (bleed hydraulics, discharge capacitors)
6. Verify zero energy state - attempt to restart
After maintenance:
1. Remove tools and replace guards
2. Verify all personnel are clear
3. Remove locks and tags (ONLY by the person who applied them)
4. Restart and test""",
        "metadata": {"type": "safety", "department": "all"},
    },
    {
        "id": "SOP-CNC-042",
        "text": """CNC MACHINE DAILY STARTUP PROCEDURE
1. Visual inspection of machine exterior and work area
2. Check coolant level - refill if below MIN line
3. Check way oil level - refill if below MIN line
4. Power on machine, home all axes
5. Run spindle warmup cycle (Program O9000): 5 min at 500 RPM, 5 min at 2000 RPM
6. Verify axis positions with test indicator
7. Check air pressure: minimum 80 PSI
8. Log startup on machine daily checklist form""",
        "metadata": {"type": "SOP", "department": "machining"},
    },
]
```

Now load them into ChromaDB:

```python
client = chromadb.Client()
collection = client.create_collection(name="manufacturing_kb")

collection.add(
    ids=[doc["id"] for doc in manufacturing_docs],
    documents=[doc["text"] for doc in manufacturing_docs],
    metadatas=[doc["metadata"] for doc in manufacturing_docs],
)
print(f"Loaded {collection.count()} documents")
```

Run it. You should see `Loaded 5 documents`. That's your knowledge base -- five manufacturing docs, embedded and searchable.

---

## Step 2: Retrieval -- Finding the Right Documents

Before we involve the LLM at all, let's just do the **retrieval** part. Ask ChromaDB to find documents relevant to our original question:

```python
results = collection.query(
    query_texts=["What's the torque spec for Assembly #4200?"],
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

It should pull back `SOP-MT-302` (the torque spec) as the top result. Notice the distance score -- lower means more relevant.

Let's try another query:

```python
results = collection.query(
    query_texts=["What PPE do I need for welding?"],
    n_results=2,
)

for doc_id in results["ids"][0]:
    print(doc_id)
```

Did it find `WPS-201`? That document mentions welding helmets, gloves, and FR clothing. ChromaDB understood that "PPE for welding" is semantically related to those safety items even though the doc never uses the letters "PPE."

That's the **Retrieve** step working. Now let's give these results to the LLM.

---

## Step 3: Augmentation -- Building the Prompt

Here's the key idea: we take the retrieved documents and stuff them into the prompt. The LLM reads them as context, then answers based on what it read.

Let's build an augmented prompt:

```python
question = "What's the torque spec for M10 bolts on Assembly #4200?"

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

Run it and look at what gets printed. Notice how the prompt now contains the actual torque spec document. The LLM won't have to guess -- the answer is right there in front of it.

---

## Step 4: Generation -- The LLM Answers with Real Data

Now send that augmented prompt to the LLM:

```python
response = llm.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": "You are a manufacturing knowledge assistant. "
            "Answer questions using only the provided documents. "
            "Be specific and cite sources.",
        },
        {"role": "user", "content": augmented_prompt},
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
```

What did you get?

It should say something like: "Per specification MT-302, M10 bolts on Frame Assembly #4200 require 45-55 Nm of torque."

**The LLM just cited a REAL spec number from YOUR documents.** That's RAG.

---

## Step 5: Side-by-Side -- Without RAG vs. With RAG

Let's see the difference clearly. Add this to your script:

```python
question = "What shielding gas should I use for MIG welding carbon steel?"

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
- Without RAG: generic answer, maybe correct in general, but no reference to YOUR welding procedure
- With RAG: cites WPS-201, mentions your specific gas mix (75% Argon / 25% CO2 at 25-30 CFH)

In manufacturing, the difference between "generally correct" and "per YOUR company spec" is everything.

---

## Step 6: What Happens When the Answer ISN'T in the Docs?

This is just as important. Try asking something your knowledge base doesn't cover:

```python
question = "What is the company's vacation policy?"

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

**That's a GOOD answer.** In manufacturing, "I don't know" is infinitely better than a hallucinated policy that could get someone in trouble. The system prompt instruction to say "I don't have that information" is your safety net.

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
                "content": "You are a manufacturing knowledge assistant. "
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
result = rag_query("How do I perform lockout/tagout?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer'][:300]}")
```

```python
result = rag_query("What's the CNC spindle warmup procedure?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer'][:300]}")
```

```python
result = rag_query("What should I do if a part fails visual inspection?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer'][:300]}")
```

Notice how the last one pulls from the QC-107 form -- "Tag with red HOLD tag, notify shift supervisor." That's a real procedure from your docs, not a generic answer.

---

## Step 8: Metadata Filtering -- Narrowing the Search

When you have hundreds of documents, you don't always want to search everything. Imagine an operator in the welding department -- they probably only need welding docs.

Let's try filtering by metadata:

```python
# Search ONLY in the quality department
results = collection.query(
    query_texts=["welding"],
    n_results=3,
    where={"department": "quality"},
)
for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
    print(f"[{doc_id}] {doc[:80]}...")
```

Without the filter, "welding" would match the welding spec WPS-201. With the quality department filter, it only returns QC-107 (the quality form that mentions weld inspection). That's more precise when you know which department you're asking about.

Try combining filters:

```python
# Search for procedures (not specs, not forms) across all departments
results = collection.query(
    query_texts=["daily procedures"],
    n_results=3,
    where={"type": "SOP"},
)
for doc_id in results["ids"][0]:
    print(doc_id)
```

This returns only SOPs, filtering out specifications and forms. Think of it like asking the filing cabinet "show me only the procedure binders, not the spec sheets."

---

## Step 9: RAG-Powered Task Description Generator

Now let's put it all together for something genuinely useful -- generating task descriptions grounded in your company's actual documents.

First, add some reference documents about your company's format and standards:

```python
# Create a fresh collection with task-focused docs
task_collection = client.create_collection(name="task_kb")

task_docs = [
    {
        "id": "ref-ppe",
        "text": "Standard PPE for welding: Auto-darkening helmet (shade 10-13), "
        "leather welding gloves, FR clothing, steel-toe boots, safety glasses under helmet.",
    },
    {
        "id": "ref-weld-inspect",
        "text": "All welding inspection per AWS D1.1 Section 6. Visual inspection criteria: "
        "no cracks, no incomplete fusion, undercut max 1/32 inch, "
        "porosity max 3/8 inch in any 12-inch length.",
    },
    {
        "id": "ref-forms",
        "text": "Quality forms: QC-107 for visual inspection, QC-110 for dimensional, "
        "QC-115 for weld-specific. All forms require inspector badge number and date.",
    },
    {
        "id": "ref-cnc-tooling",
        "text": "CNC tooling: Use only pre-approved inserts from approved vendor list (AVL). "
        "Carbide inserts for steel, CBN for hardened steel, PCD for aluminum. "
        "Log all tool changes in MES.",
    },
    {
        "id": "ref-format",
        "text": "Task description format: Title in ALL CAPS, 3-5 numbered steps starting "
        "with action verbs, safety note at bottom, reference to applicable forms/specs.",
    },
    {
        "id": "ref-forklift",
        "text": "Forklift certification required per OSHA 1910.178. "
        "Recertification every 3 years. Daily pre-operation inspection required. "
        "Speed limit: 5 MPH in production areas, 3 MPH near pedestrians.",
    },
]

task_collection.add(
    ids=[d["id"] for d in task_docs],
    documents=[d["text"] for d in task_docs],
)
print(f"Loaded {task_collection.count()} reference docs")
```

Now build the task generator:

```python
def generate_task(task_name: str) -> str:
    """Generate a task description grounded in company docs."""

    results = task_collection.query(query_texts=[task_name], n_results=3)
    context = "\n".join(
        f"- [{doc_id}]: {doc}"
        for doc_id, doc in zip(results["ids"][0], results["documents"][0])
    )

    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {
                "role": "system",
                "content": "You are a manufacturing technical writer. "
                "Generate task descriptions using the company reference documents. "
                "Follow the company format exactly. "
                "Include specific form numbers, spec references, and PPE "
                "from the reference docs. "
                "Do NOT make up form numbers or specs -- use ONLY what's provided.",
            },
            {
                "role": "user",
                "content": f'Generate a task description for: "{task_name}"\n\n'
                f"Company reference documents:\n{context}",
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content
```

Try it:

```python
print(generate_task("Perform visual inspection of MIG weld on steel frame"))
```

Look at the output. Does it reference form QC-107? QC-115? Does it mention AWS D1.1 Section 6?

**Those are REAL form numbers and spec references from YOUR documents.** The LLM didn't make them up -- it read them from the context we provided.

Try a couple more:

```python
print("=" * 60)
print(generate_task("Set up CNC lathe for aluminum shaft machining"))
```

```python
print("=" * 60)
print(generate_task("Conduct daily forklift pre-operation check"))
```

Notice how each task description pulls different reference docs. The welding task gets PPE and inspection refs. The CNC task gets tooling refs. The forklift task gets the OSHA certification info. RAG finds the right context for each task.

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

That last point is the segue into Module 07. Our basic retrieval works, but it's not perfect. What happens when the search pulls back the wrong documents? Let's find out.
