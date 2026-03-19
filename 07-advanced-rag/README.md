# Module 07: Advanced RAG Techniques

## Goal
Level up your RAG pipeline with techniques that dramatically improve retrieval quality: hybrid search, re-ranking, query transformation, and advanced chunking strategies.

---

## Concepts

### The Problem with Basic RAG

Basic RAG (Module 06) has failure modes:

| Problem | Example | Solution |
|---------|---------|----------|
| **Wrong chunks retrieved** | Query about "torque specs" retrieves "torque wrench calibration" doc | Better chunking, re-ranking |
| **Keyword mismatch** | "LOTO procedure" doesn't match "lockout/tagout" | Hybrid search |
| **Vague query** | "What do I need to know?" | Query transformation |
| **Missing context** | Chunk doesn't include the section header | Contextual chunking |
| **Too many/few results** | Retrieves 10 documents but only 2 are relevant | Re-ranking |

### The Advanced RAG Toolkit

```
           ┌──────────────┐
           │  User Query   │
           └──────┬───────┘
                  ↓
         ┌────────────────┐
         │ Query Transform│  ← Rewrite, expand, decompose
         └────────┬───────┘
                  ↓
    ┌─────────────┴──────────────┐
    ↓                            ↓
┌──────────┐              ┌──────────┐
│ Semantic  │              │ Keyword  │  ← Hybrid Search
│ Search    │              │ Search   │
└─────┬────┘              └────┬─────┘
      └──────────┬─────────────┘
                 ↓
         ┌──────────────┐
         │  Re-ranker    │  ← Score and filter
         └──────┬───────┘
                ↓
         ┌──────────────┐
         │   Generate    │
         └──────────────┘
```

---

## Exercise 1: Hybrid Search

```python
# 07-advanced-rag/ex1_hybrid_search.py
"""Combine semantic search with keyword search for better retrieval."""

import chromadb
import re
from collections import defaultdict

client = chromadb.Client()
collection = client.create_collection(name="hybrid_demo")

docs = [
    {"id": "1", "text": "LOTO procedure: Always perform lockout/tagout before servicing any equipment. Follow OSHA 1910.147 requirements."},
    {"id": "2", "text": "The hydraulic press requires monthly cylinder seal inspection. Check for oil leaks around rod seals and piston seals."},
    {"id": "3", "text": "Torque specifications for Frame #4200: M8=25Nm, M10=45Nm, M12=80Nm. Use calibrated torque wrench per SOP-MT-302."},
    {"id": "4", "text": "Personal Protective Equipment requirements: Safety glasses at all times in production. Hearing protection above 85dB. Steel-toe boots required."},
    {"id": "5", "text": "Energy isolation procedure for the stamping press: Disconnect main breaker, bleed hydraulic accumulators, lock pneumatic supply valve."},
]

collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
)


def keyword_search(query: str, documents: list[dict], top_k: int = 3) -> list[tuple[str, float]]:
    """Simple BM25-style keyword search."""
    query_terms = set(query.lower().split())
    scores = []

    for doc in documents:
        doc_terms = set(doc["text"].lower().split())
        # Simple term overlap score
        overlap = len(query_terms & doc_terms)
        score = overlap / max(len(query_terms), 1)
        scores.append((doc["id"], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def hybrid_search(query: str, top_k: int = 3, semantic_weight: float = 0.7) -> list[dict]:
    """Combine semantic and keyword search with weighted scoring."""

    # Semantic search
    semantic_results = collection.query(query_texts=[query], n_results=top_k)
    semantic_scores = {}
    for doc_id, distance in zip(semantic_results["ids"][0], semantic_results["distances"][0]):
        semantic_scores[doc_id] = 1 - distance  # Convert distance to similarity

    # Keyword search
    keyword_results = keyword_search(query, docs, top_k)
    keyword_scores = {doc_id: score for doc_id, score in keyword_results}

    # Combine scores
    all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    combined = []
    for doc_id in all_ids:
        sem = semantic_scores.get(doc_id, 0)
        kw = keyword_scores.get(doc_id, 0)
        combined_score = (semantic_weight * sem) + ((1 - semantic_weight) * kw)
        combined.append({
            "id": doc_id,
            "semantic_score": round(sem, 3),
            "keyword_score": round(kw, 3),
            "combined_score": round(combined_score, 3),
        })

    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    return combined[:top_k]


# Test: "LOTO" is an acronym — keyword search knows it, semantic search might not
query = "LOTO procedure for stamping press"
print(f"Query: '{query}'\n")

print("=== Semantic Only ===")
results = collection.query(query_texts=[query], n_results=3)
for doc_id, dist in zip(results["ids"][0], results["distances"][0]):
    doc_text = next(d["text"] for d in docs if d["id"] == doc_id)
    print(f"  [{doc_id}] (sim: {1-dist:.3f}) {doc_text[:70]}...")

print("\n=== Hybrid (70% semantic, 30% keyword) ===")
hybrid = hybrid_search(query)
for r in hybrid:
    doc_text = next(d["text"] for d in docs if d["id"] == r["id"])
    print(f"  [{r['id']}] (combined: {r['combined_score']:.3f}, sem: {r['semantic_score']:.3f}, kw: {r['keyword_score']:.3f})")
    print(f"         {doc_text[:70]}...")

print("\n=== Key Insight ===")
print("'LOTO' is jargon — keyword search catches the exact match,")
print("semantic search understands the meaning of 'energy isolation.'")
print("Hybrid search gets BOTH relevant documents.")
```

---

## Exercise 2: Query Transformation

```python
# 07-advanced-rag/ex2_query_transform.py
"""Transform user queries for better retrieval."""

import ollama
import chromadb
import json

client = chromadb.Client()
collection = client.create_collection(name="query_transform_demo")

# Load docs
docs = [
    "Torque spec MT-302: M8 bolts = 25-30 Nm on Frame Assembly #4200",
    "Weld inspection per AWS D1.1: check for cracks, porosity, undercut",
    "CNC lathe daily startup: warmup spindle 10 min, check coolant, verify axes",
    "Forklift pre-op: check tires, horn, lights, hydraulics, brakes",
    "Paint booth: maintain 65-80°F, humidity below 50%, 8-10 mils wet film",
    "PPE requirements: safety glasses always, hearing protection above 85dB",
]

collection.add(
    ids=[f"doc-{i}" for i in range(len(docs))],
    documents=docs,
)


def expand_query(original_query: str) -> list[str]:
    """Use LLM to generate multiple search queries from one question."""
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "Generate 3 different search queries that could help answer "
                "the user's question. Return as a JSON array of strings. "
                "Include the original query, a more specific version, and a version "
                "using synonyms or related terms.",
            },
            {"role": "user", "content": original_query},
        ],
        format="json",
        options={"temperature": 0.3},
    )

    try:
        result = json.loads(response["message"]["content"])
        # Handle different JSON structures
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in result:
                if isinstance(result[key], list):
                    return result[key]
        return [original_query]
    except json.JSONDecodeError:
        return [original_query]


def rag_with_query_expansion(question: str) -> dict:
    """RAG pipeline with query expansion for better retrieval."""

    # Step 1: Expand the query
    queries = expand_query(question)
    print(f"  Expanded queries: {queries}")

    # Step 2: Search with each query and combine results
    all_results = {}
    for q in queries:
        results = collection.query(query_texts=[q], n_results=2)
        for doc_id, doc, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ):
            if doc_id not in all_results or dist < all_results[doc_id]["distance"]:
                all_results[doc_id] = {"doc": doc, "distance": dist}

    # Step 3: Sort by relevance and take top results
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["distance"])
    top_docs = [r[1]["doc"] for r in sorted_results[:3]]

    # Step 4: Generate answer
    context = "\n".join(f"- {doc}" for doc in top_docs)
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "user", "content": f"Based on:\n{context}\n\nAnswer: {question}"},
        ],
        options={"temperature": 0.0},
    )

    return {
        "answer": response["message"]["content"],
        "queries_used": queries,
        "docs_retrieved": len(top_docs),
    }


# Test with vague queries that benefit from expansion
test_queries = [
    "What do I need to know before starting my shift?",
    "How tight should the bolts be?",
    "What's the painting process?",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"Original query: '{q}'")
    result = rag_with_query_expansion(q)
    print(f"Answer: {result['answer'][:200]}...")
```

---

## Exercise 3: Re-ranking Retrieved Documents

```python
# 07-advanced-rag/ex3_reranking.py
"""Re-rank retrieved documents using LLM scoring for better precision."""

import ollama
import chromadb
import json

client = chromadb.Client()
collection = client.create_collection(name="rerank_demo")

# More documents = more opportunity for retrieval mistakes
docs = [
    "Torque wrench calibration: Send to metrology lab every 6 months per SOP-CAL-001",
    "Torque specifications MT-302: M8=25-30Nm, M10=45-55Nm on Frame #4200",
    "Torque wrench types: beam, click, electronic. Click type preferred for production.",
    "Frame Assembly #4200 drawing revision history: Rev A (2022), Rev B (2023), Rev C (2024)",
    "Quality hold procedure: If torque values out of spec, quarantine all parts from that lot",
    "Operator training: Torque wrench usage covered in Module 3 of assembly training",
    "Assembly line station 5 handles Frame #4200 torque operations on second shift",
]

collection.add(
    ids=[f"doc-{i}" for i in range(len(docs))],
    documents=docs,
)


def rerank_with_llm(query: str, documents: list[str], top_k: int = 3) -> list[dict]:
    """Use the LLM to score document relevance to the query."""
    scored = []

    for i, doc in enumerate(documents):
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "system",
                    "content": "Rate how relevant the DOCUMENT is to the QUERY on a scale of 0-10. "
                    "10 = directly answers the query. 0 = completely irrelevant. "
                    "Return ONLY a JSON object: {\"score\": <number>, \"reason\": \"brief reason\"}",
                },
                {
                    "role": "user",
                    "content": f"QUERY: {query}\nDOCUMENT: {doc}",
                },
            ],
            format="json",
            options={"temperature": 0.0},
        )

        try:
            result = json.loads(response["message"]["content"])
            scored.append({
                "doc": doc,
                "score": result.get("score", 0),
                "reason": result.get("reason", ""),
            })
        except json.JSONDecodeError:
            scored.append({"doc": doc, "score": 0, "reason": "parse error"})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


query = "What are the torque specs for Frame #4200?"

# Step 1: Initial retrieval (get more than we need)
print(f"Query: '{query}'\n")
results = collection.query(query_texts=[query], n_results=5)

print("=== Initial Retrieval (top 5) ===")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  (dist: {dist:.3f}) {doc}")

# Step 2: Re-rank
print("\n=== After Re-ranking (top 3) ===")
reranked = rerank_with_llm(query, results["documents"][0])
for r in reranked:
    print(f"  (score: {r['score']}/10) {r['doc']}")
    print(f"    Reason: {r['reason']}")

print("\n=== Key Insight ===")
print("Initial retrieval returned docs about torque wrench calibration")
print("and training — related but not what was asked.")
print("Re-ranking pushed the actual spec to the top.")
print("\nTradeoff: Re-ranking adds latency (one LLM call per doc).")
print("Retrieve 5-10 docs, re-rank to top 2-3. Good compromise.")
```

---

## Takeaways

1. **Hybrid search** combines the best of keyword and semantic search — especially important for jargon and acronyms
2. **Query transformation** turns vague questions into specific searches — the LLM helps you search better
3. **Re-ranking** filters noisy retrieval results — retrieve many, re-rank to few
4. **Each technique adds latency** — use them where quality improvement justifies the cost
5. **These improvements are measurable** — Module 10 will show you exactly how much each technique helps

## Setting the Stage for Module 08

Your RAG pipeline can search and retrieve text. But real manufacturing documents come as **PDFs, Word docs, scanned images, and spreadsheets**. Module 08 teaches you to ingest, parse, and prepare real-world documents for your RAG pipeline.
