# Module 07: Advanced RAG Techniques

## Starting with a Failure

Before we improve anything, let's see where basic RAG breaks down. Set up a quick knowledge base:

```bash
touch 07-advanced-rag/advanced_rag_workshop.py
```

```python
import chromadb
from openai import OpenAI
import json

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

client = chromadb.Client()
collection = client.create_collection(name="adv_rag_demo")

docs = [
    {"id": "QB-101", "text": "Pocket passer with elite accuracy. Completes 68% of passes with a 2.3-second average release time. Excels on intermediate routes (15-25 yards) with anticipation throws. Reads defenses pre-snap and adjusts protection. Arm strength measured at 62 mph at the combine. Weakness: locks onto first read under heavy pressure."},
    {"id": "RB-201", "text": "Explosive runner with 4.38 40-yard dash. Exceptional vision through traffic and finds cutback lanes consistently. Averages 3.8 yards after contact per carry. Reliable pass catcher out of the backfield with 45 receptions last season. Weakness: needs to improve pass protection and blitz pickup."},
    {"id": "WR-301", "text": "Crisp route runner with elite separation at the top of routes. Runs the full route tree from slot and outside. 4.42 speed with a 38-inch vertical leap. Reliable hands with a 2.1% drop rate. Weakness: struggles against physical press coverage at the line of scrimmage."},
    {"id": "OL-401", "text": "Excellent anchor in pass protection with quick lateral movement to mirror speed rushers. 34-inch arm length provides leverage advantage. Run blocking grade: 82.5 out of 100. Allowed only 2 sacks in 580 pass-blocking snaps last season. Weakness: combo blocks to the second level."},
    {"id": "DEF-501", "text": "Cover-3 base defense with single-high safety. Corners play press technique on early downs. Linebackers run pattern-match zone on 3rd and long. Aggressive blitz from nickel and dime personnel. Tendency: susceptible to crossing routes against zone coverage."},
    {"id": "SPEC-601", "text": "Punter averages 46.2 yards per punt with 4.1-second hang time. Directional kicking grade: elite. Coffin corner accuracy: 73% inside the 10-yard line. Kickoff specialist reaches the end zone on 88% of attempts. Coverage units allow 7.2 average return yards."},
]

collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
)
print(f"Loaded {collection.count()} docs")
```

Now try this query:

```python
query = "good passer who can read defenses"
results = collection.query(query_texts=[query], n_results=3)

print(f"Query: '{query}'\n")
for doc_id, doc, dist in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
    print(f"  [{doc_id}] (distance: {dist:.3f}) {doc[:80]}...")
```

What happened? Depending on the embedding model, you might see:
- The QB report (QB-101) shows up -- it mentions reading defenses pre-snap
- The OL report (OL-401) might show up -- it mentions pass protection
- The DEF report (DEF-501) might appear -- it mentions pattern-match zone coverage

The problem: "passer" relates to passing, but so does "pass protection" and "pass coverage." Semantic search connects them all. And the vague term "good" doesn't help narrow things down.

**Our basic RAG got confused. Let's make it smarter.**

---

## Technique 1: Hybrid Search (Keyword + Semantic)

The idea: semantic search understands meaning ("passer" relates to "completion percentage"), but keyword search catches exact matches ("QB" literally appears in QB-101). Let's combine them.

First, build a simple keyword search:

```python
def keyword_search(query: str, documents: list[dict], top_k: int = 3) -> list[tuple[str, float]]:
    """Simple keyword overlap search."""
    query_terms = set(query.lower().split())
    scores = []

    for doc in documents:
        doc_terms = set(doc["text"].lower().split())
        overlap = len(query_terms & doc_terms)
        score = overlap / max(len(query_terms), 1)
        scores.append((doc["id"], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

Let's see what keyword search finds vs. semantic search:

```python
query = "QB accuracy and release time"

print("=== Keyword Search ===")
kw_results = keyword_search(query, docs)
for doc_id, score in kw_results:
    text = next(d["text"] for d in docs if d["id"] == doc_id)
    print(f"  [{doc_id}] (score: {score:.2f}) {text[:70]}...")

print("\n=== Semantic Search ===")
sem_results = collection.query(query_texts=[query], n_results=3)
for doc_id, dist in zip(sem_results["ids"][0], sem_results["distances"][0]):
    text = next(d["text"] for d in docs if d["id"] == doc_id)
    print(f"  [{doc_id}] (sim: {1-dist:.3f}) {text[:70]}...")
```

Notice the difference? Keyword search finds "release time" because it's a literal match in QB-101. Semantic search might also pull in passing-related docs from other positions. Neither alone gets the full picture.

Now let's combine them:

```python
def hybrid_search(query: str, top_k: int = 3, semantic_weight: float = 0.7) -> list[dict]:
    """Combine semantic and keyword search."""

    # Semantic scores
    sem_results = collection.query(query_texts=[query], n_results=top_k + 2)
    semantic_scores = {}
    for doc_id, dist in zip(sem_results["ids"][0], sem_results["distances"][0]):
        semantic_scores[doc_id] = 1 - dist

    # Keyword scores
    kw_results = keyword_search(query, docs, top_k + 2)
    keyword_scores = {doc_id: score for doc_id, score in kw_results}

    # Combine
    all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    combined = []
    for doc_id in all_ids:
        sem = semantic_scores.get(doc_id, 0)
        kw = keyword_scores.get(doc_id, 0)
        score = (semantic_weight * sem) + ((1 - semantic_weight) * kw)
        combined.append({
            "id": doc_id,
            "semantic": round(sem, 3),
            "keyword": round(kw, 3),
            "combined": round(score, 3),
        })

    combined.sort(key=lambda x: x["combined"], reverse=True)
    return combined[:top_k]
```

Try it:

```python
print("=== Hybrid Search (70% semantic, 30% keyword) ===")
results = hybrid_search("QB accuracy and release time")
for r in results:
    text = next(d["text"] for d in docs if d["id"] == r["id"])
    print(f"  [{r['id']}] combined={r['combined']:.3f} "
          f"(sem={r['semantic']:.3f}, kw={r['keyword']:.3f})")
    print(f"         {text[:70]}...")
```

Now try adjusting the weight. What happens with 50/50?

```python
results = hybrid_search("QB accuracy and release time", semantic_weight=0.5)
for r in results:
    print(f"  [{r['id']}] combined={r['combined']:.3f}")
```

Play with the `semantic_weight` parameter. For football scouting, where you have lots of position abbreviations (QB, RB, WR, OL, CB, LB) and specific metrics, keyword matching matters more than in general text. A weight of 0.5-0.7 for semantic is usually a good starting point.

---

## Technique 2: Query Expansion

Sometimes the user's query is vague or uses different words than the documents. The trick: use the LLM to rewrite the query into multiple versions, then search with all of them.

```python
def expand_query(original: str) -> list[str]:
    """Use LLM to generate multiple search queries from one question."""
    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {
                "role": "system",
                "content": "Generate 3 different search queries to help answer "
                "the user's question. Return a JSON array of strings. "
                "Include: the original query, a more specific version, "
                "and a version using synonyms or related terms.",
            },
            {"role": "user", "content": original},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    try:
        result = json.loads(response.choices[0].message.content)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in result:
                if isinstance(result[key], list):
                    return result[key]
        return [original]
    except json.JSONDecodeError:
        return [original]
```

Let's see what it does with a vague query:

```python
vague = "Who's a good passer?"
expanded = expand_query(vague)
print(f"Original: '{vague}'")
print(f"Expanded: {expanded}")
```

The LLM might generate something like:
- "Who's a good passer?"
- "Quarterback completion percentage and accuracy"
- "Passing arm strength release time anticipation throws"

Each version searches from a different angle. Let's see how this improves retrieval:

```python
# Search with original only
print("=== Original query only ===")
results = collection.query(query_texts=["Who's a good passer?"], n_results=3)
for doc_id in results["ids"][0]:
    print(f"  {doc_id}")

# Search with all expanded queries
print("\n=== After query expansion ===")
all_results = {}
for q in expanded:
    results = collection.query(query_texts=[q], n_results=2)
    for doc_id, doc, dist in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        if doc_id not in all_results or dist < all_results[doc_id]["dist"]:
            all_results[doc_id] = {"doc": doc, "dist": dist}

sorted_results = sorted(all_results.items(), key=lambda x: x[1]["dist"])
for doc_id, info in sorted_results[:3]:
    print(f"  [{doc_id}] (dist: {info['dist']:.3f}) {info['doc'][:70]}...")
```

Query expansion casts a wider net. The original vague question might miss the QB scouting report, but one of the expanded queries ("quarterback completion percentage and accuracy") hits it directly.

Now let's wrap this into a RAG function with expansion:

```python
def rag_with_expansion(question: str) -> dict:
    """RAG pipeline with query expansion."""

    queries = expand_query(question)
    print(f"  Expanded to: {queries}")

    # Search with all queries, keep best results
    all_results = {}
    for q in queries:
        results = collection.query(query_texts=[q], n_results=2)
        for doc_id, doc, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ):
            if doc_id not in all_results or dist < all_results[doc_id]["dist"]:
                all_results[doc_id] = {"doc": doc, "dist": dist}

    top_docs = sorted(all_results.items(), key=lambda x: x[1]["dist"])[:3]
    context = "\n".join(f"- {doc_id}: {info['doc']}" for doc_id, info in top_docs)

    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {"role": "user", "content": f"Based on:\n{context}\n\nAnswer: {question}"},
        ],
        temperature=0.0,
    )

    return {
        "answer": response.choices[0].message.content,
        "queries_used": queries,
        "sources": [doc_id for doc_id, _ in top_docs],
    }
```

Try it with vague questions:

```python
result = rag_with_expansion("Who's the best deep threat?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer'][:300]}")
```

```python
result = rag_with_expansion("Who's got the best hands?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer'][:300]}")
```

That last one is interesting -- "best hands" could mean catching ability (WR-301 with 2.1% drop rate) or even the RB's receiving stats. Does the expanded query help narrow it down? Check the output.

---

## Technique 3: Re-ranking

Here's the scenario: you retrieve 5 documents, but only 1 or 2 actually answer the question. The others are topically related but not useful. Re-ranking uses the LLM to score each document's relevance and keeps only the best.

Let's see the problem first:

```python
query = "How does the offensive line grade in pass protection?"
results = collection.query(query_texts=[query], n_results=5)

print(f"Query: '{query}'\n")
print("=== Raw retrieval (top 5) ===")
for doc_id, doc, dist in zip(
    results["ids"][0], results["documents"][0], results["distances"][0]
):
    print(f"  [{doc_id}] (dist: {dist:.3f}) {doc[:70]}...")
```

You'll likely see the actual OL scouting report (OL-401) but also the QB report (QB-101 mentions "adjusts protection"), the DEF scheme report (DEF-501 mentions coverage), and maybe the RB report (mentions pass protection weakness). If we feed all five to the LLM, the noise dilutes the signal.

Build the re-ranker:

```python
def rerank(query: str, documents: list[str], top_k: int = 3) -> list[dict]:
    """Use LLM to score document relevance."""
    scored = []

    for doc in documents:
        response = llm.chat.completions.create(
            model="gemma3:12b",
            messages=[
                {
                    "role": "system",
                    "content": "Rate how relevant the DOCUMENT is to answering "
                    "the QUERY. Score 0-10 where 10 = directly answers "
                    "the query. Return JSON: {\"score\": <number>, \"reason\": \"...\"}",
                },
                {
                    "role": "user",
                    "content": f"QUERY: {query}\nDOCUMENT: {doc}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        try:
            result = json.loads(response.choices[0].message.content)
            scored.append({
                "doc": doc,
                "score": result.get("score", 0),
                "reason": result.get("reason", ""),
            })
        except json.JSONDecodeError:
            scored.append({"doc": doc, "score": 0, "reason": "parse error"})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
```

Now re-rank those 5 results:

```python
print("\n=== After re-ranking (top 3) ===")
reranked = rerank(query, results["documents"][0])
for r in reranked:
    print(f"  (score: {r['score']}/10) {r['doc'][:70]}...")
    print(f"    Reason: {r['reason']}")
```

What happened? The actual OL scouting report should now be clearly at the top with a score of 9 or 10. The QB report and defensive scheme report should score much lower -- they mention pass-related concepts, but they don't answer the question about offensive line pass protection grading.

**The tradeoff**: re-ranking costs one LLM call per document. If you retrieve 5 docs, that's 5 extra LLM calls. That's why we retrieve a larger set first (cheap vector search), then re-rank to a smaller set (expensive but precise).

```python
# Time it to see the cost
import time

start = time.time()
collection.query(query_texts=[query], n_results=5)
retrieval_time = time.time() - start

start = time.time()
rerank(query, results["documents"][0], top_k=3)
rerank_time = time.time() - start

print(f"\nRetrieval: {retrieval_time:.2f}s")
print(f"Re-ranking: {rerank_time:.2f}s")
print(f"Re-ranking is ~{rerank_time/max(retrieval_time, 0.001):.0f}x slower")
```

That latency hit is real. Use re-ranking when accuracy matters more than speed -- like when building a final draft board where getting the right prospect evaluation is critical.

---

## Combining All Three Techniques

Now let's build a pipeline that uses all three: hybrid search, query expansion, and re-ranking. Each layer makes the retrieval smarter.

```python
def advanced_rag(question: str) -> dict:
    """Full advanced RAG: expand -> hybrid search -> re-rank -> generate."""

    # Step 1: Query expansion
    queries = expand_query(question)
    print(f"  1. Expanded to {len(queries)} queries")

    # Step 2: Hybrid search with each expanded query
    all_results = {}
    for q in queries:
        # Semantic
        sem = collection.query(query_texts=[q], n_results=3)
        for doc_id, doc, dist in zip(sem["ids"][0], sem["documents"][0], sem["distances"][0]):
            sem_score = 1 - dist
            # Keyword
            kw_score = len(set(q.lower().split()) & set(doc.lower().split())) / max(len(q.split()), 1)
            combined = 0.7 * sem_score + 0.3 * kw_score

            if doc_id not in all_results or combined > all_results[doc_id]["score"]:
                all_results[doc_id] = {"doc": doc, "score": combined}

    # Take top 5 candidates
    candidates = sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
    print(f"  2. Hybrid search found {len(candidates)} candidates")

    # Step 3: Re-rank to top 2
    candidate_docs = [info["doc"] for _, info in candidates]
    reranked = rerank(question, candidate_docs, top_k=2)
    print(f"  3. Re-ranked to top {len(reranked)}")

    # Step 4: Generate
    context = "\n\n".join(f"- {r['doc']}" for r in reranked)
    response = llm.chat.completions.create(
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

    return {
        "answer": response.choices[0].message.content,
        "queries_expanded": queries,
        "docs_reranked": len(reranked),
    }
```

Let's test it head-to-head with basic RAG:

```python
question = "Who's the best pass catcher in this draft class?"

print("=== Basic RAG ===")
basic = collection.query(query_texts=[question], n_results=2)
basic_context = "\n".join(basic["documents"][0])
basic_response = llm.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": f"Based on:\n{basic_context}\n\nAnswer: {question}"}],
    temperature=0.0,
)
print(f"Answer: {basic_response.choices[0].message.content[:300]}")

print("\n=== Advanced RAG ===")
advanced = advanced_rag(question)
print(f"Answer: {advanced['answer'][:300]}")
```

The advanced pipeline should give a better answer because:
1. Query expansion generated variations like "receiving ability drop rate receptions hands"
2. Hybrid search caught position-specific terms via keyword match AND catching concepts via semantic match
3. Re-ranking kept only the two most relevant docs, filtering out noise

Try a few more to compare:

```python
for q in ["Who has the fastest 40 time?", "What are the defensive tendencies on third down?"]:
    print(f"\n{'=' * 60}")
    print(f"Question: {q}")
    result = advanced_rag(q)
    print(f"Answer: {result['answer'][:200]}")
```

---

## When to Use What

Not every query needs all three techniques. Here's a practical guide:

| Technique | Use When | Cost |
|-----------|----------|------|
| **Hybrid search** | Your docs have abbreviations, prospect IDs, football jargon | Nearly free -- just keyword matching |
| **Query expansion** | Scouts ask vague or ambiguous questions | 1 extra LLM call |
| **Re-ranking** | You need high precision (final draft board, trade analysis) | 1 LLM call per candidate doc |
| **All three** | Draft day system where accuracy is critical | Multiple LLM calls, higher latency |

For a football scouting report system, hybrid search is almost always worth it (position abbreviations and prospect IDs are everywhere). Query expansion helps when scouts phrase things casually. Re-ranking is worth the cost for high-stakes draft board decisions.

## Key Takeaways

- **Hybrid search** catches what semantic search misses -- abbreviations like QB, prospect IDs like RB-201
- **Query expansion** turns vague questions into specific searches -- the LLM helps you search better
- **Re-ranking** filters noisy results -- retrieve many (cheap), keep the best (precise)
- **Each technique adds latency** -- use them where the quality improvement justifies the cost
- **Start with hybrid search** -- it's the biggest bang for the buck in football scouting contexts

Next up: your RAG pipeline is smart, but it's only as good as the documents you feed it. Module 08 tackles the messy reality of processing real PDFs, Word docs, and other formats your scouting department actually uses.
