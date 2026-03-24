# Module 05: Embeddings & Vector Stores

## What You'll Learn

How computers understand the *meaning* of text, not just the letters. By the end, you'll have a searchable database of NFL scouting reports that finds results by what they mean, not just what words they contain.

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

a = embed("Elite pocket passer with a quick release")
b = embed("Accurate quarterback who gets the ball out fast")
```

These sentences mean almost the same thing, right? Any football scout would tell you they're describing the same trait. Let's see if the numbers agree.

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
c = embed("Season ticket pricing and stadium parking availability")
print(cosine_sim(a, c))
```

Much lower -- maybe 0.2 or 0.3. The model knows "elite pocket passer" has nothing to do with "season ticket pricing."

---

## Part 3: A Football Scouting Similarity Experiment

Let's play a guessing game. Before you run each comparison, predict: will it be high (>0.7), medium (0.4-0.7), or low (<0.4)?

```python
sentences = {
    "qb_accuracy": "Quarterback with elite accuracy on intermediate throws",
    "qb_precision": "Passer who delivers the ball with pinpoint precision downfield",
    "rb_vision": "Running back with exceptional vision and patience at the line",
    "unrelated": "Season ticket pricing and stadium parking availability",
    "wr_speed": "Wide receiver with 4.3 speed and deep threat ability",
    "wr_fast": "Speedy pass catcher who stretches the field vertically",
}

vecs = {name: embed(text) for name, text in sentences.items()}
```

Now compare these pairs. Predict first, then check:

**Pair 1:** "QB accuracy" vs "QB precision"

What's your prediction? ____

```python
print(f"qb_accuracy vs qb_precision: {cosine_sim(vecs['qb_accuracy'], vecs['qb_precision']):.3f}")
```

**Pair 2:** "QB accuracy" vs "Unrelated"

Your prediction? ____

```python
print(f"qb_accuracy vs unrelated: {cosine_sim(vecs['qb_accuracy'], vecs['unrelated']):.3f}")
```

**Pair 3:** "WR speed" vs "WR fast"

Your prediction? ____

```python
print(f"wr_speed vs wr_fast: {cosine_sim(vecs['wr_speed'], vecs['wr_fast']):.3f}")
```

**Pair 4:** "RB vision" vs "WR speed"

This is the tricky one -- both are player evaluations, but different positions and skills. Your prediction? ____

```python
print(f"rb_vision vs wr_speed: {cosine_sim(vecs['rb_vision'], vecs['wr_speed']):.3f}")
```

Notice how the model captures nuance. The QB sentences are very similar. The WR sentences (different wording, same meaning) are very similar. The RB and WR sentences are somewhat related (both are player evaluations) but not as close. And season tickets are far from everything else.

**This is how semantic search works.** Instead of matching keywords, you match meaning.

---

## Part 4: Why This Beats Keyword Search

Here's a concrete example. Imagine you have a scouting report about a QB's arm talent and someone searches for "who's a good deep ball passer."

With keyword search (Ctrl+F):
- Search for "who's a good deep ball passer" -- no exact match
- Search for "passer" -- matches, but also matches "pass catcher" and "pass protection"
- Search for "deep" -- matches, but also matches "deep threat" in a WR report

With embedding search:
```python
doc = embed("Pocket passer with elite accuracy. Completes 68% of passes with a 2.3-second average release time. Excels on intermediate routes (15-25 yards) with anticipation throws.")
query = embed("who's a good deep ball passer")
print(f"Similarity: {cosine_sim(doc, query):.3f}")
```

High similarity. The search understands that the question is about the QB scouting report, even though the exact phrase never appears.

Try another:

```python
doc2 = embed("Explosive runner with 4.38 40-yard dash. Exceptional vision through traffic and finds cutback lanes consistently.")
print(f"RB doc vs QB query: {cosine_sim(doc2, query):.3f}")
```

Low similarity. It knows the running back report is irrelevant to a passing question.

---

## Part 5: Meet ChromaDB -- Your Vector Database

Comparing individual pairs is great for understanding, but in practice you need to search across hundreds or thousands of documents. That's what a vector database does.

ChromaDB is an open-source vector database that runs locally. Let's set it up:

```python
import chromadb

client = chromadb.Client()
```

That's it. No server to install, no configuration. It's running in memory.

Now create a "collection" -- think of it like building your team's film library, but for scouting report embeddings:

```python
collection = client.create_collection(name="scouting_reports")
```

Let's add our first document:

```python
collection.add(
    ids=["QB-101"],
    documents=["Pocket passer with elite accuracy. Completes 68% of passes with a 2.3-second average release time. Excels on intermediate routes (15-25 yards) with anticipation throws. Reads defenses pre-snap and adjusts protection. Arm strength measured at 62 mph at the combine. Weakness: locks onto first read under heavy pressure."],
    metadatas=[{"position": "QB", "report_type": "scouting"}],
)
```

Check that it's in there:

```python
print(f"Documents in collection: {collection.count()}")
```

Should say 1. Now let's search:

```python
results = collection.query(query_texts=["who has the strongest arm?"], n_results=1)
print(results["documents"][0][0][:80])
```

It found it. Even though we asked "who has the strongest arm?" and the document says "Arm strength measured at 62 mph at the combine" -- completely different words, same meaning.

---

## Part 6: Building Up Your Document Store

Let's add more documents one at a time and see how search improves. After each one, we'll query to see what happens.

```python
collection.add(
    ids=["RB-201"],
    documents=["Explosive runner with 4.38 40-yard dash. Exceptional vision through traffic and finds cutback lanes consistently. Averages 3.8 yards after contact per carry. Reliable pass catcher out of the backfield with 45 receptions last season. Weakness: needs to improve pass protection and blitz pickup."],
    metadatas=[{"position": "RB", "report_type": "scouting"}],
)
print(f"Total docs: {collection.count()}")
```

Query:

```python
results = collection.query(query_texts=["who is the best runner"], n_results=1)
print(results["documents"][0][0][:80])
print(results["metadatas"][0][0])
```

Good -- it found the RB doc. Now add a receiver report:

```python
collection.add(
    ids=["WR-301"],
    documents=["Crisp route runner with elite separation at the top of routes. Runs the full route tree from slot and outside. 4.42 speed with a 38-inch vertical leap. Reliable hands with a 2.1% drop rate. Weakness: struggles against physical press coverage at the line of scrimmage."],
    metadatas=[{"position": "WR", "report_type": "scouting"}],
)
```

And an offensive line report:

```python
collection.add(
    ids=["OL-401"],
    documents=["Excellent anchor in pass protection with quick lateral movement to mirror speed rushers. 34-inch arm length provides leverage advantage. Run blocking grade: 82.5 out of 100. Allowed only 2 sacks in 580 pass-blocking snaps last season. Weakness: combo blocks to the second level."],
    metadatas=[{"position": "OL", "report_type": "scouting"}],
)
```

A defensive scheme report:

```python
collection.add(
    ids=["DEF-501"],
    documents=["Cover-3 base defense with single-high safety. Corners play press technique on early downs. Linebackers run pattern-match zone on 3rd and long. Aggressive blitz from nickel and dime personnel. Tendency: susceptible to crossing routes against zone coverage."],
    metadatas=[{"position": "DEF", "report_type": "scheme"}],
)
```

And one more -- a special teams report:

```python
collection.add(
    ids=["SPEC-601"],
    documents=["Punter averages 46.2 yards per punt with 4.1-second hang time. Directional kicking grade: elite. Coffin corner accuracy: 73% inside the 10-yard line. Kickoff specialist reaches the end zone on 88% of attempts. Coverage units allow 7.2 average return yards."],
    metadatas=[{"position": "SPEC", "report_type": "scouting"}],
)

print(f"Total docs: {collection.count()}")
```

Now you have 6 documents. Let's run some searches. Before each one, predict which document it will find:

```python
queries = [
    "Who has the strongest arm in this draft class?",
    "Which prospect is the best pass catcher?",
    "What are the coverage tendencies on third down?",
    "Who grades out best in pass protection?",
    "What are this receiver's measurables?",
]

for q in queries:
    results = collection.query(query_texts=[q], n_results=2)
    print(f"\nQuery: '{q}'")
    for i in range(2):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"  #{i+1} [{meta['position']:>4}] (distance: {dist:.3f}) {doc[:70]}...")
```

Notice something interesting with "Which prospect is the best pass catcher?" -- the document doesn't mention "pass catcher" directly for the WR, but the RB report does mention "Reliable pass catcher out of the backfield." The model understands the semantic nuance. That's semantic search at work.

---

## Part 7: Metadata Filtering

Sometimes you don't want to search everything. Maybe you only want scouting reports, or only reports for a specific position.

```python
# Only search scouting reports
results = collection.query(
    query_texts=["who has the best measurables"],
    n_results=3,
    where={"report_type": "scouting"},
)

print("Scouting reports matching 'who has the best measurables':")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['position']}] {doc[:70]}...")
```

Try filtering by position:

```python
results = collection.query(
    query_texts=["pass coverage tendencies"],
    n_results=2,
    where={"report_type": "scheme"},
)

print("\nScheme reports matching 'pass coverage tendencies':")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['position']}] {doc[:70]}...")
```

You can also combine filters:

```python
results = collection.query(
    query_texts=["speed and athleticism"],
    n_results=3,
    where={"report_type": "scouting"},
)

print("\nScouting reports matching 'speed and athleticism':")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['position']}] {doc[:70]}...")
```

Metadata filtering is crucial for real systems. If a scout asks about defensive schemes, you probably don't want to return special teams reports even if they're somewhat semantically similar.

---

## Part 8: Chunking -- Why Document Size Matters

Here's a problem you'll hit immediately with real documents. Scouting reports can be pages long. If you embed the whole thing as one chunk, the embedding has to represent ALL of that information in 768 numbers. Details get lost.

Let's see this in action. Here's a comprehensive QB scouting report:

```python
long_report = """COMPREHENSIVE DRAFT SCOUTING REPORT: QUARTERBACK EVALUATION

1. ARM TALENT
Strong arm with measured velocity of 62 mph at the combine. Can make every NFL throw including the deep out and the skinny post from the far hash. Shows good touch on deep balls, placing them over the receiver's outside shoulder. Velocity on short and intermediate throws is elite, with tight spirals even under duress.

2. ACCURACY AND BALL PLACEMENT
68% completion rate over three college seasons. Best in class on intermediate routes (15-25 yards) where he completes 74% of attempts. Anticipation throws to the sideline are a strength. Accuracy declines on deep shots beyond 30 yards, completing only 41% of attempts. Ball placement on back-shoulder fades needs refinement.

3. POCKET PRESENCE
Comfortable in a collapsing pocket and navigates pressure with subtle movements. Climbs the pocket naturally when edge pressure closes in. Average time to throw: 2.3 seconds, among the fastest in this draft class. Under clean protection, completes 78% of passes. Under pressure, completion rate drops to 51%.

4. DECISION MAKING AND PROCESSING
Reads the full field on play-action concepts but tends to lock onto his first read on quick-game passes. Interception-worthy play rate: 2.4%, slightly above average. Excels in the RPO game with correct read rate of 89%. Needs to improve reading Cover-2 rotations post-snap.

5. MOBILITY AND ATHLETICISM
4.72 40-yard dash at the combine. Not a dynamic runner but extends plays when the pocket breaks down. Rushed for 342 yards and 5 touchdowns last season. Runs a controlled scramble style. Slides well and protects himself in the open field.

6. LEADERSHIP AND INTANGIBLES
Team captain for two consecutive seasons. First to arrive, last to leave per coaches. Film study habits graded as elite by coaching staff. Commands the huddle with confidence. 4-1 record in games decided by one score or less."""
```

First, let's embed the whole thing as one chunk and search:

```python
# Reset -- create a fresh collection
client = chromadb.Client()
whole_doc = client.create_collection(name="whole_doc")

whole_doc.add(ids=["full"], documents=[long_report])

results = whole_doc.query(query_texts=["What is his completion rate under pressure?"], n_results=1)
print("Whole doc result:")
print(results["documents"][0][0][:100], "...")
print(f"Distance: {results['distances'][0][0]:.3f}")
```

It returns the entire document. You get the answer somewhere in there, but also a lot of irrelevant sections. Now let's chunk it by section:

```python
import re

sections = re.split(r'\n(?=\d+\.)', long_report.strip())
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
        metadatas=[{"section_index": i, "source": "QB-EVAL-001"}],
    )
```

Search the same question:

```python
results = chunked_doc.query(query_texts=["What is his completion rate under pressure?"], n_results=1)
print("Chunked result:")
print(results["documents"][0][0])
print(f"Distance: {results['distances'][0][0]:.3f}")
```

See the difference? Instead of the whole document, you get just the "POCKET PRESENCE" section -- the exact part that answers the question. And the distance score is probably lower (meaning closer match), because the embedding is focused on pocket presence specifically.

Let's test a few more queries against both approaches:

```python
test_queries = [
    "How does he handle pre-snap reads?",
    "What are his leadership qualities?",
    "How fast is he in the 40-yard dash?",
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

Section-based chunking worked great for that scouting report because it had clear numbered sections. But not all documents are that clean. Let's compare three strategies:

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

Let's see how each one splits our scouting report:

```python
fixed = chunk_fixed(long_report)
sections = chunk_sections(long_report)
paragraphs = chunk_paragraphs(long_report)

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

Notice how fixed-size chunks cut right through the middle of sections. Chunk 1 might end in the middle of "ACCURACY AND BALL PLACEMENT" and chunk 2 starts mid-sentence. That's bad -- the embedding for that chunk is a mix of two different evaluation areas.

Section-based keeps each topic together. For structured scouting documents (player evaluations, scheme breakdowns, game reports), this is almost always the best approach.

**Rule of thumb for scouting docs:**

| Document Type | Best Chunking |
|---|---|
| Player evaluations with numbered sections | Section-based |
| Game film breakdowns | Section-based |
| Scheme/playbook analysis | Chapter/section-based |
| Weekly scout team notes | Entry-based (by date) |
| Unstructured notes/emails | Fixed-size with overlap |

---

## Part 10: Full Exercise -- Build and Query a Football Scouting Knowledge Base

Time to put it all together. Save this as `05-embeddings-and-vectors/knowledge_base.py`:

```python
"""Build a searchable NFL scouting knowledge base with ChromaDB."""

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
kb = client.create_collection(name="scouting_kb")

# --- NFL scouting documents ---
docs = [
    {
        "id": "QB-101",
        "text": "Pocket passer with elite accuracy. Completes 68% of passes with a 2.3-second "
                "average release time. Excels on intermediate routes (15-25 yards) with anticipation "
                "throws. Reads defenses pre-snap and adjusts protection. Arm strength measured at "
                "62 mph at the combine. Weakness: locks onto first read under heavy pressure.",
        "meta": {"position": "QB", "report_type": "scouting"},
    },
    {
        "id": "RB-201",
        "text": "Explosive runner with 4.38 40-yard dash. Exceptional vision through traffic and "
                "finds cutback lanes consistently. Averages 3.8 yards after contact per carry. "
                "Reliable pass catcher out of the backfield with 45 receptions last season. "
                "Weakness: needs to improve pass protection and blitz pickup.",
        "meta": {"position": "RB", "report_type": "scouting"},
    },
    {
        "id": "WR-301",
        "text": "Crisp route runner with elite separation at the top of routes. Runs the full "
                "route tree from slot and outside. 4.42 speed with a 38-inch vertical leap. "
                "Reliable hands with a 2.1% drop rate. Weakness: struggles against physical press "
                "coverage at the line of scrimmage.",
        "meta": {"position": "WR", "report_type": "scouting"},
    },
    {
        "id": "OL-401",
        "text": "Excellent anchor in pass protection with quick lateral movement to mirror speed "
                "rushers. 34-inch arm length provides leverage advantage. Run blocking grade: 82.5 "
                "out of 100. Allowed only 2 sacks in 580 pass-blocking snaps last season. "
                "Weakness: combo blocks to the second level.",
        "meta": {"position": "OL", "report_type": "scouting"},
    },
    {
        "id": "DEF-501",
        "text": "Cover-3 base defense with single-high safety. Corners play press technique on "
                "early downs. Linebackers run pattern-match zone on 3rd and long. Aggressive blitz "
                "from nickel and dime personnel. Tendency: susceptible to crossing routes against "
                "zone coverage.",
        "meta": {"position": "DEF", "report_type": "scheme"},
    },
    {
        "id": "SPEC-601",
        "text": "Punter averages 46.2 yards per punt with 4.1-second hang time. Directional "
                "kicking grade: elite. Coffin corner accuracy: 73% inside the 10-yard line. "
                "Kickoff specialist reaches the end zone on 88% of attempts. Coverage units "
                "allow 7.2 average return yards.",
        "meta": {"position": "SPEC", "report_type": "scouting"},
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
    "Who has the strongest arm in this draft class?",
    "Which prospect is the best pass catcher?",
    "What are the coverage tendencies on third down?",
    "Who grades out best in pass protection?",
    "What are this receiver's measurables?",
]

for query in queries:
    results = kb.query(query_texts=[query], n_results=2)
    print(f"\nQ: {query}")
    for i in range(2):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        print(f"  #{i+1} [{meta['position']:>4} | {meta['report_type']:<8}] "
              f"(dist: {dist:.3f}) {doc[:65]}...")

# --- Filtered search ---
print(f"\n{'=' * 60}")
print("FILTERED SEARCH: Scouting reports only")
print("=" * 60)

results = kb.query(
    query_texts=["who has the best measurables"],
    n_results=3,
    where={"report_type": "scouting"},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['position']}] {doc[:70]}...")

# --- Embedding comparison ---
print(f"\n{'=' * 60}")
print("EMBEDDING SIMILARITY DEMO")
print("=" * 60)

pairs = [
    ("Elite pocket passer with quick release", "Accurate QB who gets the ball out fast"),
    ("Elite pocket passer with quick release", "Season ticket pricing and parking"),
    ("Route runner with great separation", "Receiver who gets open consistently"),
    ("Pass blocking technique", "Run blocking scheme"),
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
- **Semantic search beats keyword search** -- "who's a good deep ball passer" matches a QB scouting report about "anticipation throws" even with zero shared words
- **ChromaDB** stores embeddings and lets you search across many documents instantly
- **Metadata filtering** narrows search to specific positions, report types, or schemes
- **Chunking matters** -- section-based chunking is best for structured scouting reports; fixed-size is a fallback for unstructured text
- **Smaller, focused chunks** produce better search results than embedding entire documents

## Up Next: Module 06

You now have the two core ingredients: structured LLM output (Module 04) and semantic document search (this module). Module 06 puts them together into a complete **RAG pipeline** -- load your documents, chunk them, embed them, search them, and feed the results to the LLM. That's the architecture behind your NFL scouting and draft analysis system.
