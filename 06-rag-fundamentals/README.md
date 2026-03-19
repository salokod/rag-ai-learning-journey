# Module 06: RAG Fundamentals

## Goal
Build a complete Retrieval-Augmented Generation pipeline from scratch. By the end, your LLM will answer questions using YOUR documents, not just its training data.

---

## Concepts

### What Is RAG?

RAG = **Retrieval**-Augmented **Generation**

Instead of asking an LLM to answer from memory (training data), you:
1. **Retrieve** relevant documents from your knowledge base
2. **Augment** the prompt with those documents
3. **Generate** an answer grounded in the retrieved context

```
Without RAG:
  User: "What's the torque spec for Assembly #4200?"
  LLM: "I don't know your specific specs." (or worse, hallucmates one)

With RAG:
  User: "What's the torque spec for Assembly #4200?"
  → System retrieves: [spec doc showing 25-30 Nm]
  → Prompt includes: "Based on these documents: [spec doc]..."
  LLM: "Per specification MT-302, Assembly #4200 requires 25-30 Nm torque."
```

### The RAG Pipeline

```
┌──────────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐
│ User Question│ ──→ │ Retrieve │ ──→ │  Augment   │ ──→ │ Generate │
│              │     │ (vector  │     │ (build     │     │ (LLM     │
│              │     │  search) │     │  prompt)   │     │  answer) │
└──────────────┘     └──────────┘     └────────────┘     └──────────┘
```

### Why RAG Instead of Fine-Tuning?

| | RAG | Fine-Tuning |
|---|---|---|
| **Data freshness** | Always up-to-date (just update docs) | Requires retraining |
| **Traceability** | Can cite sources | Black box |
| **Setup effort** | Hours | Days to weeks |
| **Best for** | Factual Q&A, reference lookup | Style/behavior changes |

For your manufacturing task descriptions: **start with RAG**. It's faster, traceable, and your SOPs change frequently.

---

## Exercise 1: Build a RAG Pipeline from Scratch

```python
# 06-rag-fundamentals/ex1_basic_rag.py
"""Build a complete RAG pipeline step by step."""

import chromadb
import ollama

# ============================================================
# STEP 1: Prepare your knowledge base
# ============================================================
print("=== Step 1: Loading Knowledge Base ===")

# These represent your company's actual documents
manufacturing_docs = [
    {
        "id": "WPS-201",
        "text": """WELDING PROCEDURE SPECIFICATION WPS-201
Process: GMAW (MIG Welding)
Base Metal: Carbon Steel (A36)
Filler Metal: ER70S-6, 0.035" diameter
Shielding Gas: 75% Argon / 25% CO2 at 25-30 CFH
Preheat: Not required for material under 1" thick
Interpass Temperature: 400°F maximum
Post-Weld: Visual inspection required. Ultrasonic testing for critical joints.
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
Failure action: Tag with red "HOLD" tag, notify shift supervisor""",
        "metadata": {"type": "form", "department": "quality"},
    },
    {
        "id": "SOP-MT-302",
        "text": """TORQUE SPECIFICATION MT-302
Assembly: Frame Assembly #4200
All fasteners: Grade 8, zinc plated
M8 bolts: 25-30 Nm
M10 bolts: 45-55 Nm
M12 bolts: 80-100 Nm
Sequence: Follow the torque pattern diagram (star pattern)
Tool: Calibrated torque wrench, ±2% accuracy
Verification: 10% sampling by QC after assembly
Documentation: Record on Form QC-110""",
        "metadata": {"type": "specification", "department": "assembly"},
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
6. Verify zero energy state — attempt to restart
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
2. Check coolant level — refill if below MIN line
3. Check way oil level — refill if below MIN line
4. Power on machine, home all axes
5. Run spindle warmup cycle (Program O9000): 5 min at 500 RPM, 5 min at 2000 RPM
6. Verify axis positions with test indicator
7. Check air pressure: minimum 80 PSI
8. Log startup on machine daily checklist form""",
        "metadata": {"type": "SOP", "department": "machining"},
    },
]

# ============================================================
# STEP 2: Create vector store and load documents
# ============================================================
print("=== Step 2: Creating Vector Store ===")

client = chromadb.Client()
collection = client.create_collection(name="manufacturing_kb")

collection.add(
    ids=[doc["id"] for doc in manufacturing_docs],
    documents=[doc["text"] for doc in manufacturing_docs],
    metadatas=[doc["metadata"] for doc in manufacturing_docs],
)
print(f"✓ Loaded {collection.count()} documents")

# ============================================================
# STEP 3: Build the RAG function
# ============================================================

def rag_query(question: str, n_results: int = 2) -> dict:
    """Complete RAG pipeline: retrieve → augment → generate."""

    # RETRIEVE: Find relevant documents
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]
    distances = results["distances"][0]

    # AUGMENT: Build the prompt with context
    context = "\n\n---\n\n".join(
        f"[Source: {doc_id}]\n{doc}"
        for doc_id, doc in zip(retrieved_ids, retrieved_docs)
    )

    augmented_prompt = f"""Answer the question based ONLY on the provided context documents.
If the answer is not in the context, say "I don't have that information in my documents."
Always cite which source document(s) you used.

CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""

    # GENERATE: Get the LLM's response
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a manufacturing knowledge assistant. Answer questions "
                "using only the provided documents. Be specific and cite sources.",
            },
            {"role": "user", "content": augmented_prompt},
        ],
        options={"temperature": 0.0},
    )

    return {
        "question": question,
        "answer": response["message"]["content"],
        "sources": retrieved_ids,
        "distances": distances,
        "context": context,
    }


# ============================================================
# STEP 4: Test it!
# ============================================================
print("\n=== Step 3-4: RAG in Action ===\n")

test_questions = [
    "What is the torque specification for M10 bolts on Assembly #4200?",
    "What shielding gas do I use for MIG welding carbon steel?",
    "How do I perform lockout/tagout?",
    "What should I do if a part fails visual inspection?",
    "What's the CNC spindle warmup procedure?",
    "What is the company's vacation policy?",  # Not in our docs!
]

for question in test_questions:
    result = rag_query(question)
    print(f"Q: {result['question']}")
    print(f"Sources: {result['sources']}")
    print(f"A: {result['answer'][:200]}...")
    print()
```

---

## Exercise 2: RAG for Task Description Generation

```python
# 06-rag-fundamentals/ex2_rag_task_generation.py
"""Use RAG to generate task descriptions grounded in actual company docs."""

import chromadb
import ollama

# Set up knowledge base (same as Exercise 1)
client = chromadb.Client()
collection = client.create_collection(name="manufacturing_kb_v2")

# Your company's reference documents
docs = [
    {"id": "ref-001", "text": "Standard PPE for welding: Auto-darkening helmet (shade 10-13), leather welding gloves, FR clothing, steel-toe boots, safety glasses under helmet."},
    {"id": "ref-002", "text": "All welding inspection per AWS D1.1 Section 6. Visual inspection criteria: no cracks, no incomplete fusion, undercut max 1/32 inch, porosity max 3/8 inch in any 12-inch length."},
    {"id": "ref-003", "text": "Quality forms: QC-107 for visual inspection, QC-110 for dimensional, QC-115 for weld-specific. All forms require inspector badge number and date."},
    {"id": "ref-004", "text": "CNC tooling: Use only pre-approved inserts from the approved vendor list (AVL). Carbide inserts for steel, CBN for hardened steel, PCD for aluminum. Log all tool changes in MES."},
    {"id": "ref-005", "text": "Existing task description format: Title in caps, 3-5 numbered steps starting with action verbs, safety note at bottom, reference to applicable forms/specs."},
    {"id": "ref-006", "text": "Forklift certification required per OSHA 1910.178. Recertification every 3 years. Daily pre-operation inspection required. Speed limit: 5 MPH in production areas, 3 MPH near pedestrians."},
]

collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
    metadatas=[{"source": d["id"]} for d in docs],
)


def generate_task_with_rag(task_name: str) -> dict:
    """Generate a task description using RAG for company-specific details."""

    # Retrieve relevant reference docs
    results = collection.query(query_texts=[task_name], n_results=3)
    context_docs = results["documents"][0]
    source_ids = results["ids"][0]

    context = "\n".join(f"- [{sid}]: {doc}" for sid, doc in zip(source_ids, context_docs))

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": """You are a manufacturing technical writer.
Generate task descriptions using the company's reference documents.
Follow the company's existing format exactly.
Include specific form numbers, spec references, and PPE from the reference docs.
Do NOT make up form numbers or specs — use ONLY what's in the reference documents.""",
            },
            {
                "role": "user",
                "content": f"""Generate a task description for: "{task_name}"

Use these company reference documents:
{context}

Write the task description now, following the company format.""",
            },
        ],
        options={"temperature": 0.1},
    )

    return {
        "task_name": task_name,
        "description": response["message"]["content"],
        "sources_used": source_ids,
    }


# Test
tasks = [
    "Perform visual inspection of MIG weld on steel frame",
    "Set up CNC lathe for aluminum shaft machining",
    "Conduct daily forklift pre-operation check",
]

for task_name in tasks:
    result = generate_task_with_rag(task_name)
    print(f"{'='*60}")
    print(f"Task: {result['task_name']}")
    print(f"Sources used: {result['sources_used']}")
    print(f"{'='*60}")
    print(result["description"])
    print()

print("=== Why This Is Better Than No RAG ===")
print("The generated descriptions reference YOUR company's actual:")
print("  - Form numbers (QC-107, QC-110, QC-115)")
print("  - Specifications (AWS D1.1 Section 6)")
print("  - PPE requirements (specific to your facility)")
print("  - Existing format conventions")
print("\nBut how do we KNOW it's actually better? Module 09-13 answers that.")
```

---

## Exercise 3: RAG Pipeline with Metadata Filtering

```python
# 06-rag-fundamentals/ex3_metadata_filtering.py
"""Use metadata to filter retrieval results for more precise RAG."""

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="filtered_kb")

# Documents with rich metadata
docs = [
    {"id": "1", "text": "Welding procedure for carbon steel frames", "meta": {"dept": "welding", "type": "procedure", "safety_level": "high"}},
    {"id": "2", "text": "Welding procedure for aluminum panels", "meta": {"dept": "welding", "type": "procedure", "safety_level": "high"}},
    {"id": "3", "text": "Weld inspection checklist for quality control", "meta": {"dept": "quality", "type": "checklist", "safety_level": "medium"}},
    {"id": "4", "text": "CNC machining setup for steel parts", "meta": {"dept": "machining", "type": "procedure", "safety_level": "medium"}},
    {"id": "5", "text": "Safety training for press brake operation", "meta": {"dept": "press", "type": "training", "safety_level": "critical"}},
    {"id": "6", "text": "Annual press brake maintenance schedule", "meta": {"dept": "press", "type": "maintenance", "safety_level": "high"}},
]

collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
    metadatas=[d["meta"] for d in docs],
)

# Search with metadata filters
print("=== Search: 'welding' in quality department only ===")
results = collection.query(
    query_texts=["welding"],
    n_results=3,
    where={"dept": "quality"},  # Only quality dept docs
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['dept']}] {doc}")

print("\n=== Search: 'safety' with critical safety level ===")
results = collection.query(
    query_texts=["safety procedures"],
    n_results=3,
    where={"safety_level": "critical"},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['safety_level']}] {doc}")

print("\n=== Search: procedures in press department ===")
results = collection.query(
    query_texts=["how to operate"],
    n_results=3,
    where={"$and": [{"dept": "press"}, {"type": {"$ne": "maintenance"}}]},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['dept']}/{meta['type']}] {doc}")

print("\n=== Why Metadata Filtering Matters ===")
print("Without filters: 'welding' returns welding AND quality docs")
print("With dept filter: only returns quality dept's welding-related docs")
print("This precision dramatically improves RAG quality for large doc sets.")
```

---

## Takeaways

1. **RAG = Retrieve + Augment + Generate** — the core pattern for grounding LLMs in your data
2. **Your documents become the LLM's knowledge** — it can reference specific form numbers, specs, and procedures
3. **"I don't have that information"** is a GOOD answer — better than hallucinating
4. **Metadata filtering** makes retrieval more precise when you have many documents
5. **This is your capstone architecture** — everything from here builds on this pipeline

## Setting the Stage for Module 07

Your basic RAG pipeline works. But "works" isn't enough for production. Module 07 covers **advanced RAG techniques** — better chunking, hybrid search (combining keyword + semantic), re-ranking retrieved documents, and query transformation. These techniques can dramatically improve retrieval quality, which directly improves answer quality.
