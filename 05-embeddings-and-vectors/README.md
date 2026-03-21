# Module 05: Embeddings & Vector Stores

## What You'll Learn

How computers understand the *meaning* of text, not just the letters. By the end, you'll have a searchable database of manufacturing documents that finds results by what they mean, not just what words they contain.

**Time:** ~75 minutes of hands-on work

**Prerequisites:** Module 04 complete, Ollama running

---

## Part 1: Pull the Embedding Model

First things first. You need a model that turns text into numbers. Run this in your terminal:

```bash
ollama pull nomic-embed-text
```

This downloads a model specifically designed to create embeddings -- numerical representations of meaning. It's small (about 274MB) and fast, even on CPU.

Once it finishes, let's make sure it works. Open Python:

```bash
python3
```

```python
from openai import OpenAI

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = llm.embeddings.create(model="nomic-embed-text", input="Hello world")
print(type(response.data[0].embedding))
print(len(response.data[0].embedding))
```

You should see a list of 768 numbers. That's the "meaning" of "Hello world" -- encoded as 768 floating-point numbers. Not very human-readable, but incredibly useful for computers.

Let's peek at the first few numbers:

```python
print(response.data[0].embedding[:10])
```

Just a bunch of decimals. Not interesting to look at directly. But here's where it gets interesting.

---

## Part 2: Similar Meaning = Similar Numbers

Let's embed two sentences that mean the same thing but use different words:

```python
def embed(text):
    return llm.embeddings.create(model="nomic-embed-text", input=text).data[0].embedding

a = embed("Inspect the weld joints for cracks")
b = embed("Check welding quality for defects")
```

These sentences mean almost the same thing, right? An experienced welder would tell you they're describing the same task. Let's see if the numbers agree.

We need a way to compare two lists of numbers. The standard method is called *cosine similarity*:

```python
import numpy as np

def cosine_sim(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

This gives a score between -1 and 1. Closer to 1 means more similar. Let's try it:

```python
print(cosine_sim(a, b))
```

What did you get? Probably something around 0.8 or higher. High similarity -- the model understands these sentences are about the same thing, even though they share almost no words.

Now let's try something completely unrelated:

```python
c = embed("Order new office supplies for the break room")
print(cosine_sim(a, c))
```

Much lower -- maybe 0.2 or 0.3. The model knows "inspecting welds" has nothing to do with "ordering office supplies."

---

## Part 3: A Manufacturing Similarity Experiment

Let's play a guessing game. Before you run each comparison, predict: will it be high (>0.7), medium (0.4-0.7), or low (<0.4)?

```python
sentences = {
    "weld_inspect": "Inspect weld joints for cracks and porosity",
    "weld_quality": "Check welding quality and structural integrity",
    "torque_check": "Verify the torque on all fastener assemblies",
    "office":       "Order new office supplies for the break room",
    "calibrate":    "Calibrate the digital pressure gauge monthly",
    "cal_rephrase": "The pressure measurement device needs regular calibration",
}

vecs = {name: embed(text) for name, text in sentences.items()}
```

Now compare these pairs. Predict first, then check:

**Pair 1:** "Inspect weld joints" vs "Check welding quality"

What's your prediction? ____

```python
print(f"weld_inspect vs weld_quality: {cosine_sim(vecs['weld_inspect'], vecs['weld_quality']):.3f}")
```

**Pair 2:** "Inspect weld joints" vs "Order office supplies"

Your prediction? ____

```python
print(f"weld_inspect vs office: {cosine_sim(vecs['weld_inspect'], vecs['office']):.3f}")
```

**Pair 3:** "Calibrate pressure gauge" vs "Pressure device needs calibration"

Your prediction? ____

```python
print(f"calibrate vs cal_rephrase: {cosine_sim(vecs['calibrate'], vecs['cal_rephrase']):.3f}")
```

**Pair 4:** "Torque on fasteners" vs "Calibrate pressure gauge"

This is the tricky one -- both are measurement/QC-related, but different tasks. Your prediction? ____

```python
print(f"torque_check vs calibrate: {cosine_sim(vecs['torque_check'], vecs['calibrate']):.3f}")
```

Notice how the model captures nuance. The welding sentences are very similar. The calibration sentences (different wording, same meaning) are very similar. The torque and calibration sentences are somewhat related (both are measurement tasks) but not as close. And office supplies is far from everything else.

**This is how semantic search works.** Instead of matching keywords, you match meaning.

---

## Part 4: Why This Beats Keyword Search

Here's a concrete example. Imagine you have a document about "TIG welding aluminum" and someone searches for "how to weld aluminum parts."

With keyword search (Ctrl+F):
- Search for "how to weld aluminum parts" -- no exact match
- Search for "weld" -- matches, but also matches "weld spatter cleanup" and "weld inspection"
- Search for "aluminum" -- matches, but also matches "aluminum storage rack inventory"

With embedding search:
```python
doc = embed("TIG welding aluminum requires argon shielding gas at 15-20 CFH")
query = embed("how to weld aluminum parts")
print(f"Similarity: {cosine_sim(doc, query):.3f}")
```

High similarity. The search understands that the question is about the welding document, even though the exact phrase never appears.

Try another:

```python
doc2 = embed("Forklift daily inspection: check tires, test horn and lights")
print(f"Forklift doc vs welding query: {cosine_sim(doc2, query):.3f}")
```

Low similarity. It knows the forklift document is irrelevant to a welding question.

---

## Part 5: Meet ChromaDB -- Your Vector Database

Comparing individual pairs is great for understanding, but in practice you need to search across hundreds or thousands of documents. That's what a vector database does.

ChromaDB is an open-source vector database that runs locally. Let's set it up:

```python
import chromadb

client = chromadb.Client()
```

That's it. No server to install, no configuration. It's running in memory.

Now create a "collection" -- think of it like a table in a regular database, but for embeddings:

```python
collection = client.create_collection(name="manufacturing_docs")
```

Let's add our first document:

```python
collection.add(
    ids=["sop-001"],
    documents=["TIG welding aluminum requires argon shielding gas at 15-20 CFH flow rate. Clean the base metal with acetone and a stainless steel brush before welding. Preheat thick sections to 300 degrees F."],
    metadatas=[{"department": "welding", "type": "SOP"}],
)
```

Check that it's in there:

```python
print(f"Documents in collection: {collection.count()}")
```

Should say 1. Now let's search:

```python
results = collection.query(query_texts=["how do I weld aluminum?"], n_results=1)
print(results["documents"][0][0][:80])
```

It found it. Even though we asked "how do I weld aluminum?" and the document says "TIG welding aluminum requires argon shielding gas" -- completely different words, same meaning.

---

## Part 6: Building Up Your Document Store

Let's add more documents one at a time and see how search improves. After each one, we'll query to see what happens.

```python
collection.add(
    ids=["sop-002"],
    documents=["CNC lathe setup: Load the program from the DNC server. Install the correct chuck jaw set per the setup sheet. Zero the X and Z axes using the tool setter. Run the first part at 50 percent feed rate override."],
    metadatas=[{"department": "machining", "type": "SOP"}],
)
print(f"Total docs: {collection.count()}")
```

Query:

```python
results = collection.query(query_texts=["how to set up a lathe"], n_results=1)
print(results["documents"][0][0][:80])
print(results["metadatas"][0][0])
```

Good -- it found the CNC doc. Now add a safety document:

```python
collection.add(
    ids=["sop-003"],
    documents=["Hydraulic press safety: Verify the light curtain is operational before each shift. Never bypass safety interlocks. Two-hand operation is required for all press operations over 5 tons. Inspect cylinder seals monthly for leaks."],
    metadatas=[{"department": "press", "type": "safety"}],
)
```

And a quality doc:

```python
collection.add(
    ids=["sop-004"],
    documents=["Incoming material inspection: Check mill certificates against PO specifications. Verify material grade, dimensions, and surface condition. Use digital calipers for dimensional checks. Flag any non-conforming material with a red tag."],
    metadatas=[{"department": "quality", "type": "SOP"}],
)
```

A finishing doc:

```python
collection.add(
    ids=["sop-005"],
    documents=["Paint booth operation: Check air filtration system before starting. Maintain booth temperature between 65-80 degrees F and humidity below 50 percent. Apply primer coat at 8-10 mils wet thickness. Allow 30-minute flash-off between coats."],
    metadatas=[{"department": "finishing", "type": "SOP"}],
)
```

And one more -- a forklift doc:

```python
collection.add(
    ids=["sop-006"],
    documents=["Forklift daily inspection checklist: Check tire condition and pressure. Test horn, lights, and backup alarm. Verify hydraulic fluid level. Check mast chains for wear. Test brakes before loading. Report any defects to maintenance before operating."],
    metadatas=[{"department": "warehouse", "type": "safety"}],
)

print(f"Total docs: {collection.count()}")
```

Now you have 6 documents. Let's run some searches. Before each one, predict which document it will find:

```python
queries = [
    "What are the safety rules for the press?",
    "How do I check incoming steel quality?",
    "What temperature should the paint booth be?",
    "How do I inspect a forklift?",
    "What gas do I use for aluminum welding?",
]

for q in queries:
    results = collection.query(query_texts=[q], n_results=2)
    print(f"\nQuery: '{q}'")
    for i in range(2):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"  #{i+1} [{meta['department']:>10}] (distance: {dist:.3f}) {doc[:70]}...")
```

Notice something interesting with "How do I check incoming steel quality?" -- the document doesn't mention "steel" at all, just "material." But the model understands that incoming steel is a type of incoming material inspection. That's semantic search at work.

---

## Part 7: Metadata Filtering

Sometimes you don't want to search everything. Maybe you only want safety documents, or only docs from the welding department.

```python
# Only search safety documents
results = collection.query(
    query_texts=["inspection procedures"],
    n_results=3,
    where={"type": "safety"},
)

print("Safety docs matching 'inspection procedures':")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['department']}] {doc[:70]}...")
```

Try filtering by department:

```python
results = collection.query(
    query_texts=["how to prepare for work"],
    n_results=2,
    where={"department": "welding"},
)

print("\nWelding dept docs matching 'how to prepare for work':")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['department']}] {doc[:70]}...")
```

You can also combine filters:

```python
results = collection.query(
    query_texts=["daily checks"],
    n_results=3,
    where={"type": "safety"},
)

print("\nSafety docs matching 'daily checks':")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['department']}] {doc[:70]}...")
```

Metadata filtering is crucial for real systems. If an operator asks about welding, you probably don't want to return forklift docs even if they're somewhat semantically similar.

---

## Part 8: Chunking -- Why Document Size Matters

Here's a problem you'll hit immediately with real documents. Manufacturing SOPs can be pages long. If you embed the whole thing as one chunk, the embedding has to represent ALL of that information in 768 numbers. Details get lost.

Let's see this in action. Here's a realistic SOP:

```python
long_sop = """STANDARD OPERATING PROCEDURE: COMPLETE CNC MACHINING WORKFLOW

1. JOB PREPARATION
Receive the work order from the scheduler. Review the engineering drawing and verify
the revision level matches the work order. Check that raw material is available in
the staging area and matches the material specification. Read all engineering notes
and special instructions.

2. MACHINE SETUP
Power on the CNC machine and allow the spindle to warm up for 10 minutes. Load the
correct program from the DNC server. Verify program number against the setup sheet.
Install the appropriate chuck or fixture per the setup sheet. Install cutting tools
in the correct turret stations per the tool list.

3. TOOL SETTING
Use the tool presetter or touch-off method to establish tool offsets. Verify each
tool offset against the setup sheet values with tolerance of plus or minus 0.001 inches.
Enter tool wear offsets as zero for new tools. Load tool life counters per the tool
management plan.

4. FIRST ARTICLE INSPECTION
Run the first part at 50 percent rapid and 75 percent feed override. After the first
part is complete, remove it and perform dimensional inspection per the inspection plan.
Record measurements on the First Article Inspection Report. If all dimensions are
within tolerance, proceed to production. If any dimension is out of tolerance, adjust
offsets and re-run.

5. PRODUCTION RUN
After first article approval, run production at programmed speeds and feeds. Monitor
the process for unusual sounds, vibrations, or chip formation. Check parts per the
sampling plan, typically every 10th part. Keep the work area clean and organized.

6. DOCUMENTATION
Complete the production traveler with actual quantities, times, and any issues.
Submit completed first article report and traveler to quality. Log machine time in
the MES system. Clean the machine and work area before the next job."""
```

First, let's embed the whole thing as one chunk and search:

```python
# Reset -- create a fresh collection
client = chromadb.Client()
whole_doc = client.create_collection(name="whole_doc")

whole_doc.add(ids=["full"], documents=[long_sop])

results = whole_doc.query(query_texts=["How do I set tool offsets?"], n_results=1)
print("Whole doc result:")
print(results["documents"][0][0][:100], "...")
print(f"Distance: {results['distances'][0][0]:.3f}")
```

It returns the entire document. You get the answer somewhere in there, but also a lot of irrelevant sections. Now let's chunk it by section:

```python
import re

sections = re.split(r'\n(?=\d+\.)', long_sop.strip())
sections = [s.strip() for s in sections if s.strip()]

print(f"Number of sections: {len(sections)}")
for i, s in enumerate(sections):
    first_line = s.split('\n')[0]
    print(f"  Section {i}: {first_line} ({len(s)} chars)")
```

Now store each section as its own chunk:

```python
chunked_doc = client.create_collection(name="chunked_doc")

for i, section in enumerate(sections):
    chunked_doc.add(
        ids=[f"section-{i}"],
        documents=[section],
        metadatas=[{"section_index": i, "source": "SOP-CNC-001"}],
    )
```

Search the same question:

```python
results = chunked_doc.query(query_texts=["How do I set tool offsets?"], n_results=1)
print("Chunked result:")
print(results["documents"][0][0])
print(f"Distance: {results['distances'][0][0]:.3f}")
```

See the difference? Instead of the whole document, you get just the "TOOL SETTING" section -- the exact part that answers the question. And the distance score is probably lower (meaning closer match), because the embedding is focused on tool setting specifically.

Let's test a few more queries against both approaches:

```python
test_queries = [
    "What do I do after machining the first part?",
    "How do I document my work?",
    "What should I check before starting the machine?",
]

for query in test_queries:
    r_whole = whole_doc.query(query_texts=[query], n_results=1)
    r_chunk = chunked_doc.query(query_texts=[query], n_results=1)

    chunk_section = r_chunk["documents"][0][0].split('\n')[0]
    print(f"\nQuery: '{query}'")
    print(f"  Whole doc distance: {r_whole['distances'][0][0]:.3f}")
    print(f"  Chunk distance:     {r_chunk['distances'][0][0]:.3f} --> {chunk_section}")
```

The chunked version should consistently have lower distances (better matches) and return the specific relevant section.

---

## Part 9: Comparing Chunking Strategies

Section-based chunking worked great for that SOP because it had clear numbered sections. But not all documents are that clean. Let's compare three strategies:

```python
# Strategy 1: Fixed-size chunks (500 chars with overlap)
def chunk_fixed(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# Strategy 2: Section-based (what we just did)
def chunk_sections(text):
    sections = re.split(r'\n(?=\d+\.)', text.strip())
    return [s.strip() for s in sections if s.strip()]

# Strategy 3: Paragraph-based
def chunk_paragraphs(text):
    paras = text.strip().split('\n\n')
    return [p.strip() for p in paras if p.strip()]
```

Let's see how each one splits our SOP:

```python
fixed = chunk_fixed(long_sop)
sections = chunk_sections(long_sop)
paragraphs = chunk_paragraphs(long_sop)

print(f"Fixed-size:  {len(fixed)} chunks")
for i, c in enumerate(fixed):
    print(f"  {i}: {len(c)} chars -- '{c[:50]}...'")

print(f"\nSection:     {len(sections)} chunks")
for i, c in enumerate(sections):
    print(f"  {i}: {len(c)} chars -- '{c.split(chr(10))[0]}'")

print(f"\nParagraph:   {len(paragraphs)} chunks")
for i, c in enumerate(paragraphs):
    print(f"  {i}: {len(c)} chars -- '{c[:50]}...'")
```

Notice how fixed-size chunks cut right through the middle of sections. Chunk 1 might end in the middle of "MACHINE SETUP" and chunk 2 starts mid-sentence. That's bad -- the embedding for that chunk is a mix of two different topics.

Section-based keeps each topic together. For structured manufacturing documents (SOPs, work instructions, checklists), this is almost always the best approach.

**Rule of thumb for manufacturing docs:**

| Document Type | Best Chunking |
|---|---|
| SOPs with numbered sections | Section-based |
| Work instructions | Section-based |
| Equipment manuals | Chapter/section-based |
| Maintenance logs | Entry-based (by date) |
| Unstructured notes/emails | Fixed-size with overlap |

---

## Part 10: Full Exercise -- Build and Query a Manufacturing Knowledge Base

Time to put it all together. Save this as `05-embeddings-and-vectors/knowledge_base.py`:

```python
"""Build a searchable manufacturing knowledge base with ChromaDB."""

import chromadb
from openai import OpenAI
import numpy as np
import re

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- Set up ChromaDB ---
client = chromadb.Client()
kb = client.create_collection(name="manufacturing_kb")

# --- Manufacturing documents ---
docs = [
    {
        "id": "weld-001",
        "text": "TIG welding aluminum requires argon shielding gas at 15-20 CFH flow rate. "
                "Clean the base metal with acetone and a stainless steel brush before welding. "
                "Preheat thick sections (over 0.25 inch) to 300 degrees F.",
        "meta": {"department": "welding", "type": "SOP", "equipment": "TIG welder"},
    },
    {
        "id": "cnc-001",
        "text": "CNC lathe setup: Load the program from the DNC server. Install the correct "
                "chuck jaw set per the setup sheet. Zero the X and Z axes using the tool setter. "
                "Run the first part at 50 percent feed rate override.",
        "meta": {"department": "machining", "type": "SOP", "equipment": "CNC lathe"},
    },
    {
        "id": "press-001",
        "text": "Hydraulic press safety: Verify the light curtain is operational before each "
                "shift. Never bypass safety interlocks. Two-hand operation is required for all "
                "press operations over 5 tons. Inspect cylinder seals monthly for leaks.",
        "meta": {"department": "press", "type": "safety", "equipment": "hydraulic press"},
    },
    {
        "id": "qa-001",
        "text": "Incoming material inspection: Check mill certificates against PO specifications. "
                "Verify material grade, dimensions, and surface condition. Use digital calipers "
                "for dimensional checks. Flag any non-conforming material with a red tag.",
        "meta": {"department": "quality", "type": "SOP", "equipment": "calipers"},
    },
    {
        "id": "paint-001",
        "text": "Paint booth operation: Check air filtration system before starting. Maintain "
                "booth temperature between 65-80 degrees F and humidity below 50 percent. Apply "
                "primer coat at 8-10 mils wet thickness. Allow 30-minute flash-off between coats.",
        "meta": {"department": "finishing", "type": "SOP", "equipment": "paint booth"},
    },
    {
        "id": "fork-001",
        "text": "Forklift daily inspection checklist: Check tire condition and pressure. Test "
                "horn, lights, and backup alarm. Verify hydraulic fluid level. Check mast chains "
                "for wear. Test brakes before loading. Report any defects to maintenance.",
        "meta": {"department": "warehouse", "type": "safety", "equipment": "forklift"},
    },
]

# Add all documents
for doc in docs:
    kb.add(ids=[doc["id"]], documents=[doc["text"]], metadatas=[doc["meta"]])

print(f"Knowledge base loaded: {kb.count()} documents\n")

# --- Interactive search demo ---
print("=" * 60)
print("SEMANTIC SEARCH DEMO")
print("=" * 60)

queries = [
    "What gas do I need for welding?",
    "How do I check if steel meets specs?",
    "What PPE or safety checks for the hydraulic press?",
    "What's the right temperature for painting?",
    "How do I start up the CNC machine?",
]

for query in queries:
    results = kb.query(query_texts=[query], n_results=2)
    print(f"\nQ: {query}")
    for i in range(2):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        print(f"  #{i+1} [{meta['department']:>10} | {meta['type']:<6}] "
              f"(dist: {dist:.3f}) {doc[:65]}...")

# --- Filtered search ---
print(f"\n{'=' * 60}")
print("FILTERED SEARCH: Safety docs only")
print("=" * 60)

results = kb.query(
    query_texts=["daily inspection checklist"],
    n_results=3,
    where={"type": "safety"},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['department']}] {doc[:70]}...")

# --- Embedding comparison ---
print(f"\n{'=' * 60}")
print("EMBEDDING SIMILARITY DEMO")
print("=" * 60)

pairs = [
    ("Inspect weld joints for cracks", "Check welding quality for defects"),
    ("Inspect weld joints for cracks", "Order office supplies"),
    ("Calibrate the pressure gauge", "The pressure meter needs calibration"),
    ("Set up the CNC lathe", "Program the milling machine"),
]

for text1, text2 in pairs:
    e1 = llm.embeddings.create(model="nomic-embed-text", input=text1).data[0].embedding
    e2 = llm.embeddings.create(model="nomic-embed-text", input=text2).data[0].embedding
    sim = cosine_sim(e1, e2)
    bar = "#" * int(sim * 30)
    print(f"  {sim:.3f} {bar}")
    print(f"    '{text1}'")
    print(f"    '{text2}'\n")
```

Run it:

```bash
cd 05-embeddings-and-vectors
python knowledge_base.py
```

---

## What You Now Know

- **Embeddings** turn text into numbers that capture meaning -- 768 numbers per sentence with nomic-embed-text
- **Cosine similarity** measures how close two embeddings are (1.0 = identical meaning, 0.0 = unrelated)
- **Semantic search beats keyword search** -- "check welding quality" matches "inspect weld joints" even with zero shared words
- **ChromaDB** stores embeddings and lets you search across many documents instantly
- **Metadata filtering** narrows search to specific departments, document types, or equipment
- **Chunking matters** -- section-based chunking is best for structured manufacturing SOPs; fixed-size is a fallback for unstructured text
- **Smaller, focused chunks** produce better search results than embedding entire documents

## Up Next: Module 06

You now have the two core ingredients: structured LLM output (Module 04) and semantic document search (this module). Module 06 puts them together into a complete **RAG pipeline** -- load your documents, chunk them, embed them, search them, and feed the results to the LLM. That's the architecture behind your manufacturing task description system.
