# Module 18: Production & Deployment

## Goal
Learn the patterns for deploying LLM applications to production: API design, caching, error handling, cost management, and monitoring. Turn your prototype into a reliable service.

---

## Concepts

### Production Concerns Checklist

```
□ API Layer      — How do users interact with your system?
□ Caching        — Don't recompute identical queries
□ Error Handling — Graceful degradation, retries, fallbacks
□ Rate Limiting  — Prevent abuse and cost overruns
□ Monitoring     — Know when things break (Module 12)
□ Cost Control   — Track and limit token usage
□ Security       — API keys, input validation, guardrails (Module 16)
□ Versioning     — Track which model/prompt version served each request
```

---

## Exercise 1: Production RAG API Server

```python
# 18-production-and-deployment/ex1_api_server.py
"""A production-quality RAG API server with caching, logging, and error handling."""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import chromadb
import ollama
import hashlib
import json
import time
import logging
from functools import lru_cache
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

app = FastAPI(
    title="Manufacturing RAG API",
    description="Task description generation with RAG",
    version="1.0.0",
)

# === Data Models ===
class QueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    department: str | None = None
    n_results: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    quality_score: float
    cached: bool
    latency_ms: float

class TaskGenerationRequest(BaseModel):
    task_name: str = Field(min_length=5, max_length=200)
    department: str
    context: str = ""

class TaskGenerationResponse(BaseModel):
    task_description: str
    sources_used: list[str]
    quality_checks: dict
    version: str

# === Simple Cache ===
class ResponseCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, **kwargs) -> str:
        content = json.dumps({"query": query, **kwargs}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> dict | None:
        key = self._key(query, **kwargs)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, query: str, response: dict, **kwargs):
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        key = self._key(query, **kwargs)
        self.cache[key] = response

cache = ResponseCache()

# === Rate Limiter ===
class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def check(self, client_id: str) -> bool:
        now = time.time()
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter()

# === Knowledge Base Setup ===
def setup_knowledge_base():
    client = chromadb.Client()
    collection = client.create_collection(name="production_kb")
    # In production, this would load from persistent storage
    docs = [
        ("MT-302", "Torque Spec MT-302: Frame #4200. M8=25-30Nm, M10=45-55Nm, M12=80-100Nm."),
        ("WPS-201", "WPS-201: GMAW carbon steel. ER70S-6. 75/25 Ar/CO2. Interpass 400°F max."),
        ("QC-107", "Form QC-107: Visual inspection. Surface, welds, hardware, coating. Fail = HOLD tag."),
        ("SAFE-001", "LOTO SOP-SAFE-001: Notify, shutdown, isolate, lock/tag, release energy, verify."),
        ("PPE-001", "PPE: Glasses always. Welding: helmet shade 10-13, gloves, FR. Grinding: face shield."),
    ]
    collection.add(
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
    )
    return collection

kb = setup_knowledge_base()

# === API Endpoints ===
@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} — {elapsed:.0f}ms")
    return response

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(req: QueryRequest, request: Request):
    """Query the manufacturing knowledge base with RAG."""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    start = time.time()

    # Check cache
    cached_response = cache.get(req.question, dept=req.department)
    if cached_response:
        return QueryResponse(**cached_response, cached=True,
                           latency_ms=(time.time() - start) * 1000)

    # RAG pipeline
    try:
        results = kb.query(query_texts=[req.question], n_results=req.n_results)
        context = "\n".join(results["documents"][0])
        source_ids = results["ids"][0]

        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": "Answer using only the context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQ: {req.question}"},
            ],
            options={"temperature": 0.0},
        )

        answer = response["message"]["content"]
        quality = min(len(answer.split()) / 30, 1.0)

        result = {
            "answer": answer,
            "sources": source_ids,
            "quality_score": round(quality, 2),
        }

        cache.set(req.question, result, dept=req.department)

        return QueryResponse(
            **result,
            cached=False,
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing query")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        ollama.list()
        return {"status": "healthy", "model": "llama3.1:8b", "kb_docs": kb.count()}
    except Exception:
        return {"status": "degraded", "error": "Ollama not reachable"}

@app.get("/stats")
async def get_stats():
    """Cache and rate limiter statistics."""
    return {
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_hit_rate": f"{cache.hits / max(cache.hits + cache.misses, 1):.1%}",
        "cache_size": len(cache.cache),
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://localhost:8000")
    print("Docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

To run: `pip install fastapi uvicorn && python 18-production-and-deployment/ex1_api_server.py`

---

## Exercise 2: Cost Tracking and Management

```python
# 18-production-and-deployment/ex2_cost_tracking.py
"""Track and manage LLM costs — even when using local models, track token usage."""

import ollama
import json
from datetime import datetime
from collections import defaultdict


class CostTracker:
    """Track token usage and costs across your LLM application."""

    # Approximate costs per 1M tokens (for when you use cloud models)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "llama3.1:8b": {"input": 0.00, "output": 0.00},  # Local = free!
    }

    def __init__(self):
        self.usage_log = []
        self.daily_totals = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})

    def log_usage(self, model: str, input_tokens: int, output_tokens: int,
                  purpose: str = "general"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "purpose": purpose,
            "estimated_cost": self._estimate_cost(model, input_tokens, output_tokens),
        }
        self.usage_log.append(entry)

        date = datetime.now().strftime("%Y-%m-%d")
        self.daily_totals[date]["input_tokens"] += input_tokens
        self.daily_totals[date]["output_tokens"] += output_tokens
        self.daily_totals[date]["calls"] += 1

        return entry

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def get_summary(self) -> dict:
        total_input = sum(e["input_tokens"] for e in self.usage_log)
        total_output = sum(e["output_tokens"] for e in self.usage_log)
        total_cost = sum(e["estimated_cost"] for e in self.usage_log)

        by_purpose = defaultdict(lambda: {"calls": 0, "tokens": 0})
        for e in self.usage_log:
            by_purpose[e["purpose"]]["calls"] += 1
            by_purpose[e["purpose"]]["tokens"] += e["input_tokens"] + e["output_tokens"]

        return {
            "total_calls": len(self.usage_log),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "estimated_cost": f"${total_cost:.4f}",
            "by_purpose": dict(by_purpose),
        }


# Demo
tracker = CostTracker()

# Simulate some queries
queries = [
    ("What's the torque spec?", "rag_query"),
    ("Write task description for weld inspection", "task_generation"),
    ("Evaluate this task description...", "evaluation"),
]

print("=== Cost Tracking Demo ===\n")
for query, purpose in queries:
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": query}],
        options={"temperature": 0.0},
    )

    # Approximate token counts
    input_tokens = len(query) // 4
    output_tokens = len(response["message"]["content"]) // 4

    entry = tracker.log_usage("llama3.1:8b", input_tokens, output_tokens, purpose)
    print(f"  {purpose}: ~{input_tokens + output_tokens} tokens (${entry['estimated_cost']:.4f})")

summary = tracker.get_summary()
print(f"\n=== Summary ===")
print(f"Total calls: {summary['total_calls']}")
print(f"Total tokens: {summary['total_tokens']}")
print(f"Estimated cost: {summary['estimated_cost']}")
print(f"By purpose: {json.dumps(summary['by_purpose'], indent=2)}")

print("\n=== Why Track Costs Even for Local Models? ===")
print("1. Know your token volume — important when you switch to cloud models")
print("2. Identify expensive operations (evaluation uses 3-5x more tokens)")
print("3. Optimize: cache common queries, reduce prompt size, batch operations")
print("4. Budget planning: 'if we move to GPT-4o, this pipeline costs $X/month'")
```

---

## Takeaways

1. **API design matters** — FastAPI gives you typed endpoints, auto-docs, and validation
2. **Cache aggressively** — identical RAG queries should return cached results
3. **Rate limit everything** — protect against abuse and cost overruns
4. **Track token usage** — even with local models, know your volume for future cloud migration
5. **Health checks and monitoring** — know when your system is degraded before users report it

## Setting the Stage for Module 19

You've learned every component. Now it's time to **build the real thing**. Module 19 is your capstone: a complete manufacturing task description system with RAG, evaluation, guardrails, observability, and production-ready deployment.
