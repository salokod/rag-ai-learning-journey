# Module 05: Embeddings & Vector Stores

## Goal
Understand how text becomes numbers (embeddings), how to search by meaning instead of keywords, and how to store/retrieve documents using a vector database. This is the retrieval half of RAG.

---

## Concepts

### What Are Embeddings?

An embedding converts text into a list of numbers (a vector) that captures its **meaning**. Similar texts produce similar vectors.

```
"Inspect weld joints"     → [0.23, -0.45, 0.87, 0.12, ...]  (768 numbers)
"Check welding quality"   → [0.21, -0.43, 0.85, 0.14, ...]  (very similar!)
"Order lunch for the team" → [-0.67, 0.34, -0.12, 0.56, ...] (very different)
```

**Why this matters:** Traditional search (Ctrl+F) finds exact text matches. Embedding search finds **meaning** matches. A search for "welding inspection" will find documents about "weld joint quality assessment" even though the words are different.

### Vector Stores

A vector store (or vector database) is a database optimized for storing and searching embeddings. Think of it as "Google for your documents, but it understands meaning."

We'll use **ChromaDB** — it's open-source, runs locally, and is perfect for learning.

### The Embedding Pipeline

```
Your Documents → Chunk into pieces → Embed each chunk → Store in vector DB
                                                              ↓
User Query → Embed the query → Find similar chunks → Return top matches
```

---

## Exercise 1: Your First Embeddings

```python
# 05-embeddings-and-vectors/ex1_first_embeddings.py
"""Create embeddings and see how similarity works."""

import ollama
import numpy as np


def get_embedding(text: str) -> list[float]:
    """Get an embedding vector from Ollama."""
    response = ollama.embed(model="nomic-embed-text", input=text)
    return response["embeddings"][0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate how similar two vectors are (1.0 = identical, 0.0 = unrelated)."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# First, pull the embedding model
print("Make sure you've run: ollama pull nomic-embed-text\n")

# Create embeddings for manufacturing-related sentences
sentences = [
    "Inspect the weld joints for cracks and porosity",
    "Check welding quality and structural integrity",
    "Verify the torque on all fastener assemblies",
    "Order new office supplies for the break room",
    "Calibrate the digital pressure gauge monthly",
    "The pressure measurement device needs regular calibration",
]

print("=== Embedding Similarity Matrix ===\n")

# Get all embeddings
embeddings = {s: get_embedding(s) for s in sentences}

# Show dimension
first_emb = list(embeddings.values())[0]
print(f"Embedding dimension: {len(first_emb)} numbers per sentence\n")

# Compare each pair
print(f"{'':>55} | Similarity")
print("-" * 75)

pairs = [
    (0, 1, "weld inspect vs weld quality (SIMILAR MEANING)"),
    (0, 3, "weld inspect vs office supplies (UNRELATED)"),
    (4, 5, "calibrate gauge vs calibration device (SIMILAR)"),
    (2, 4, "torque vs pressure gauge (SOMEWHAT RELATED)"),
]

for i, j, label in pairs:
    sim = cosine_similarity(embeddings[sentences[i]], embeddings[sentences[j]])
    bar = "█" * int(sim * 40)
    print(f"  {label:>50} | {sim:.3f} {bar}")

print("\n=== Key Insight ===")
print("Embeddings capture MEANING, not just keywords.")
print("'Inspect weld joints' and 'Check welding quality' are similar")
print("even though they share almost no words!")
```

---

## Exercise 2: Building a Vector Store with ChromaDB

```python
# 05-embeddings-and-vectors/ex2_chromadb_basics.py
"""Build a searchable document store with ChromaDB."""

import chromadb
import ollama

# Create a local ChromaDB instance (data stored in memory for now)
client = chromadb.Client()

# Create a collection (like a table in a regular database)
collection = client.create_collection(
    name="manufacturing_procedures",
    metadata={"description": "Manufacturing SOPs and procedures"},
)

# Sample manufacturing documents
documents = [
    {
        "id": "sop-001",
        "text": "TIG welding aluminum requires argon shielding gas at 15-20 CFH flow rate. "
        "Clean the base metal with acetone and a stainless steel brush before welding. "
        "Preheat thick sections (>0.25 inch) to 300°F.",
        "metadata": {"department": "welding", "type": "SOP"},
    },
    {
        "id": "sop-002",
        "text": "CNC lathe setup procedure: Load the program from the DNC server. "
        "Install the correct chuck jaw set per the setup sheet. Zero the X and Z axes "
        "using the tool setter. Run the first part at 50% feed rate override.",
        "metadata": {"department": "machining", "type": "SOP"},
    },
    {
        "id": "sop-003",
        "text": "Hydraulic press safety: Verify the light curtain is operational before "
        "each shift. Never bypass safety interlocks. Two-hand operation is required "
        "for all press operations over 5 tons. Inspect cylinder seals monthly for leaks.",
        "metadata": {"department": "press", "type": "safety"},
    },
    {
        "id": "sop-004",
        "text": "Incoming material inspection: Check mill certificates against PO specifications. "
        "Verify material grade, dimensions, and surface condition. Use digital calipers "
        "for dimensional checks. Flag any non-conforming material with a red tag.",
        "metadata": {"department": "quality", "type": "SOP"},
    },
    {
        "id": "sop-005",
        "text": "Paint booth operation: Check air filtration system before starting. "
        "Maintain booth temperature between 65-80°F and humidity below 50%. "
        "Apply primer coat at 8-10 mils wet thickness. Allow 30-minute flash-off "
        "between coats.",
        "metadata": {"department": "finishing", "type": "SOP"},
    },
    {
        "id": "sop-006",
        "text": "Forklift daily inspection checklist: Check tire condition and pressure. "
        "Test horn, lights, and backup alarm. Verify hydraulic fluid level. "
        "Check mast chains for wear. Test brakes before loading. Report any defects "
        "to maintenance before operating.",
        "metadata": {"department": "warehouse", "type": "safety"},
    },
]

# Add documents to the collection
# ChromaDB will automatically create embeddings using its default model
collection.add(
    ids=[doc["id"] for doc in documents],
    documents=[doc["text"] for doc in documents],
    metadatas=[doc["metadata"] for doc in documents],
)

print(f"✓ Added {collection.count()} documents to the vector store\n")

# Now search by MEANING
queries = [
    "How do I weld aluminum?",
    "What are the safety rules for the press?",
    "How do I check incoming steel quality?",
    "What temperature should the paint booth be?",
]

for query in queries:
    print(f"Query: '{query}'")
    results = collection.query(query_texts=[query], n_results=2)

    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        relevance = 1 - dist  # Convert distance to relevance score
        print(f"  #{i+1} [{meta['department']:>10}] (relevance: {relevance:.2f}) {doc[:80]}...")
    print()

print("=== Key Insight ===")
print("The query 'How do I check incoming steel quality?' found the")
print("incoming material inspection doc — even though 'steel' isn't in that doc.")
print("That's the power of semantic search.")
```

---

## Exercise 3: Chunking Strategies

```python
# 05-embeddings-and-vectors/ex3_chunking.py
"""Learn why and how to break documents into chunks."""

import chromadb

# A longer manufacturing document
long_document = """
STANDARD OPERATING PROCEDURE: COMPLETE CNC MACHINING WORKFLOW

1. JOB PREPARATION
Receive the work order from the scheduler. Review the engineering drawing and verify
the revision level matches the work order. Check that raw material is available in
the staging area and matches the material specification. Read all engineering notes
and special instructions.

2. MACHINE SETUP
Power on the CNC machine and allow the spindle to warm up for 10 minutes. Load the
correct program from the DNC server — verify program number against the setup sheet.
Install the appropriate chuck or fixture per the setup sheet. Install cutting tools
in the correct turret stations per the tool list.

3. TOOL SETTING
Use the tool presetter or touch-off method to establish tool offsets. Verify each
tool offset against the setup sheet values (tolerance: ±0.001"). Enter tool wear
offsets as zero for new tools. Load tool life counters per the tool management plan.

4. FIRST ARTICLE INSPECTION
Run the first part at 50% rapid and 75% feed override. After the first part is
complete, remove it and perform dimensional inspection per the inspection plan.
Record measurements on the First Article Inspection Report (FAIR). If all
dimensions are within tolerance, proceed to production. If any dimension is out
of tolerance, adjust offsets and re-run.

5. PRODUCTION RUN
After FAIR approval, run production at programmed speeds and feeds. Monitor the
process for unusual sounds, vibrations, or chip formation. Check parts per the
sampling plan (typically every 10th part). Keep the work area clean and organized.

6. DOCUMENTATION
Complete the production traveler with actual quantities, times, and any issues.
Submit completed FAIR and traveler to quality. Log machine time in the MES system.
Clean the machine and work area before the next job.
"""

print("=== Full Document ===")
print(f"Total characters: {len(long_document)}")
print(f"Approx tokens: {len(long_document) // 4}")

# Chunking Strategy 1: Fixed-size chunks
def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# Chunking Strategy 2: Section-based chunks
def chunk_by_sections(text: str) -> list[str]:
    """Split text by numbered sections."""
    import re
    sections = re.split(r'\n(?=\d+\.)', text.strip())
    return [s.strip() for s in sections if s.strip()]

# Chunking Strategy 3: Paragraph-based
def chunk_by_paragraphs(text: str) -> list[str]:
    """Split text by double newlines (paragraphs)."""
    paragraphs = text.strip().split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]

print("\n=== Strategy 1: Fixed Size (500 chars, 100 overlap) ===")
fixed = chunk_fixed_size(long_document)
for i, chunk in enumerate(fixed):
    print(f"  Chunk {i}: {len(chunk)} chars — '{chunk[:60]}...'")

print(f"\n=== Strategy 2: By Section ===")
sections = chunk_by_sections(long_document)
for i, chunk in enumerate(sections):
    first_line = chunk.split("\n")[0]
    print(f"  Chunk {i}: {len(chunk)} chars — '{first_line}'")

print(f"\n=== Strategy 3: By Paragraph ===")
paragraphs = chunk_by_paragraphs(long_document)
for i, chunk in enumerate(paragraphs):
    print(f"  Chunk {i}: {len(chunk)} chars — '{chunk[:60]}...'")

print("\n=== Which Strategy to Use? ===")
print("Section-based  — BEST for structured docs (SOPs, manuals) like yours")
print("Fixed-size     — Good fallback when docs have no clear structure")
print("Paragraph      — Good for unstructured text (emails, notes)")
print("\nChunking HUGELY affects RAG quality. We'll evaluate this in Module 07.")

# Store section-based chunks and test retrieval
client = chromadb.Client()
collection = client.create_collection(name="cnc_procedure")

for i, chunk in enumerate(sections):
    collection.add(
        ids=[f"section-{i}"],
        documents=[chunk],
        metadatas=[{"section_index": i, "source": "SOP-CNC-001"}],
    )

# Test: Can it find the right section?
test_queries = [
    "How do I set up tools on the CNC?",
    "What do I do after machining the first part?",
    "How do I document my work?",
]

print("\n=== Retrieval Test ===")
for query in test_queries:
    results = collection.query(query_texts=[query], n_results=1)
    doc = results["documents"][0][0]
    first_line = doc.split("\n")[0]
    print(f"  Q: {query}")
    print(f"  A: Found section: '{first_line}'")
    print()
```

---

## Takeaways

1. **Embeddings capture meaning** — similar concepts produce similar vectors, regardless of exact wording
2. **ChromaDB** is your local vector store — open-source, no server setup needed
3. **Chunking is critical** — bad chunking = bad retrieval = bad RAG. Section-based is best for structured docs
4. **Overlap in chunking** prevents losing context at chunk boundaries
5. **Semantic search >> keyword search** for finding relevant manufacturing procedures

## Setting the Stage for Module 06

You can now embed documents and search by meaning. Module 06 puts it all together into a complete **RAG pipeline**: load documents → chunk → embed → store → retrieve → generate. This is the core architecture you'll use in your manufacturing task description system.
