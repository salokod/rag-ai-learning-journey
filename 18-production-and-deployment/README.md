# Module 18: Production & Deployment

## You can build it. Can you SHIP it?

You've got a RAG pipeline that generates manufacturing task descriptions. It retrieves relevant docs, generates grounded output, evaluates quality, and catches hallucinations.

But right now it only runs in a script on your laptop.

In this module, you'll turn it into something you could actually hand to another team: a real API server with caching, rate limiting, health checks, cost tracking, and graceful error handling. All running locally on your M4 Pro.

---

## Step 1: Your First API Endpoint

Let's get a server running. We're going to use FastAPI because it gives you typed endpoints, automatic docs, and validation for free.

Create this file:

```python
# 18-production-and-deployment/step1_basic_api.py
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Manufacturing RAG API")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

That's 9 lines. Run it:

```bash
pip install fastapi uvicorn
python 18-production-and-deployment/step1_basic_api.py
```

You should see something like:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Now, in a **second terminal**, hit it:

```bash
curl http://localhost:8000/health
```

You should get back:

```json
{"status": "healthy"}
```

You just built and served an HTTP API. That `/health` endpoint? That's what monitoring systems (Kubernetes, load balancers, uptime checkers) ping to know your service is alive.

Now try this:

```bash
curl http://localhost:8000/docs
```

Or just open http://localhost:8000/docs in your browser. FastAPI generates interactive API documentation automatically. Every endpoint you add shows up there with a "Try it out" button.

Stop the server with `Ctrl+C`. Let's make it do something useful.

---

## Step 2: Add a Knowledge Base and RAG Endpoint

Let's wire up ChromaDB and Ollama. First, the knowledge base setup:

```python
# 18-production-and-deployment/step2_rag_api.py
from fastapi import FastAPI
from openai import OpenAI
import chromadb
import uvicorn

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

app = FastAPI(title="Manufacturing RAG API")
```

Now add the knowledge base. In production, you'd load this from persistent storage. For now, we'll seed it inline:

```python
def setup_kb():
    client = chromadb.Client()
    collection = client.create_collection(name="production_kb")
    docs = [
        ("MT-302", "Torque Spec MT-302: Frame #4200. M8=25-30Nm, M10=45-55Nm, M12=80-100Nm."),
        ("WPS-201", "WPS-201: GMAW carbon steel. ER70S-6. 75/25 Ar/CO2. Interpass 400F max."),
        ("QC-107", "Form QC-107: Visual inspection. Surface, welds, hardware, coating. Fail = HOLD tag."),
        ("SAFE-001", "LOTO SOP-SAFE-001: Notify, shutdown, isolate, lock/tag, release energy, verify."),
        ("PPE-001", "PPE: Glasses always. Welding: helmet shade 10-13, gloves, FR. Grinding: face shield."),
    ]
    collection.add(
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
    )
    return collection

kb = setup_kb()
```

Think of this like stocking the parts crib before a shift. You need your reference materials loaded before anyone asks a question.

Now the query endpoint. We need a request model first:

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    n_results: int = Field(default=3, ge=1, le=10)
```

Notice the validation built right in. `min_length=5` means nobody can send you an empty string. `ge=1, le=10` means n_results must be between 1 and 10. FastAPI enforces this automatically and returns a clear error if the input is bad.

Now the actual endpoint:

```python
@app.post("/query")
async def query_kb(req: QueryRequest):
    results = kb.query(query_texts=[req.question], n_results=req.n_results)
    context = "\n".join(results["documents"][0])
    source_ids = results["ids"][0]

    response = llm.chat.completions.create(
        model="llama3.3:70b",
        messages=[
            {"role": "system", "content": "Answer using only the context. Cite sources."},
            {"role": "user", "content": f"Context:\n{context}\n\nQ: {req.question}"},
        ],
        temperature=0.0,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": source_ids,
    }
```

Put it all together in one file and run it:

```bash
python 18-production-and-deployment/step2_rag_api.py
```

Now query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What PPE do I need for welding?"}'
```

You just served an LLM-powered RAG API locally. A user sends a question, your server retrieves relevant docs from the vector store, sends them to the LLM, and returns a grounded answer with source citations.

Notice how long that took? Probably a few seconds. Let's fix that for repeated queries.

---

## Step 3: Add Caching

Same question twice? Don't recompute. That LLM call is the expensive part -- why do it again for an identical query?

Here's a simple cache class:

```python
import hashlib
import json

class ResponseCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, query, **kwargs):
        content = json.dumps({"query": query, **kwargs}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query, **kwargs):
        key = self._key(query, **kwargs)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, query, response, **kwargs):
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[self._key(query, **kwargs)] = response
```

The idea: hash the query into a key, store the result. Next time we see the same query, return the stored result instead of calling the LLM.

This is exactly like keeping a log of part numbers you've already inspected. If the same part comes through again and nothing has changed, you don't re-inspect it from scratch.

Now wire it into the endpoint. Add this right before the LLM call:

```python
cache = ResponseCache()

@app.post("/query")
async def query_kb(req: QueryRequest):
    # Check cache first
    cached = cache.get(req.question, n=req.n_results)
    if cached:
        return {**cached, "cached": True}

    # ... existing RAG code ...

    result = {"answer": answer, "sources": source_ids}
    cache.set(req.question, result, n=req.n_results)
    return {**result, "cached": False}
```

Restart the server and test it:

```bash
# First call -- cache miss, hits the LLM
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What PPE do I need for welding?"}'

# Same call again -- should be instant
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What PPE do I need for welding?"}'
```

Notice the difference? The second call should come back nearly instantly. Look at the response -- it should say `"cached": true`. That's a cache hit. No LLM call, no vector search, just a dictionary lookup.

---

## Step 4: Add Rate Limiting

What if someone writes a script that hammers your API 1,000 times per second? That's going to choke your LLM, burn through resources, and slow things down for everyone.

Rate limiting means: each client gets N requests per time window. After that, they get a 429 "Too Many Requests" error.

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=60, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def check(self, client_id):
        now = time.time()
        # Remove old timestamps outside the window
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True
```

Wire it in:

```python
from fastapi import HTTPException, Request

rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

@app.post("/query")
async def query_kb(req: QueryRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    # ... rest of the endpoint ...
```

Let's set a low limit (10 per minute) so you can actually test it. Restart and try:

```bash
# Run this in a loop -- after 10, you'll get 429s
for i in $(seq 1 12); do
  echo "Request $i:"
  curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What PPE do I need for welding?"}' | python3 -m json.tool | head -3
  echo
done
```

After request 10, you should see: `"detail": "Rate limit exceeded. Try again shortly."`

That's your guardrail against abuse. In production, you'd set this higher (maybe 100/min) and use an API key instead of IP address for client identification.

---

## Step 5: A Real Health Check

The `/health` endpoint we built earlier is too simple. A real health check should verify that your dependencies are actually working. If Ollama is down, your API is effectively useless even though FastAPI itself is running.

```python
@app.get("/health")
async def health_check():
    checks = {}

    # Can we reach Ollama?
    try:
        llm.models.list()
        checks["ollama"] = "ok"
    except Exception as e:
        checks["ollama"] = f"error: {e}"

    # Is our knowledge base loaded?
    try:
        count = kb.count()
        checks["knowledge_base"] = f"ok ({count} docs)"
    except Exception as e:
        checks["knowledge_base"] = f"error: {e}"

    all_ok = all(v.startswith("ok") for v in checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
    }
```

Restart and test:

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

You should see both checks passing. Now try this: stop Ollama, then hit health again:

```bash
# Stop Ollama
killall ollama

# Check health
curl http://localhost:8000/health | python3 -m json.tool
```

Notice the status says "degraded" instead of "healthy". That's exactly what a monitoring system needs. It can alert you before users start reporting errors.

Start Ollama back up:

```bash
ollama serve &
```

---

## Step 6: Cost and Token Tracking

Even with local models (cost = $0), you should track token volume. Why?

1. When you eventually test cloud models, you'll know exactly what it'll cost
2. You can spot expensive operations (evaluation uses 3-5x more tokens than generation)
3. You can budget: "if we move to GPT-4o, this pipeline costs $X/month"

```python
from datetime import datetime

class CostTracker:
    PRICING_PER_1M = {
        "llama3.3:70b": {"input": 0.00, "output": 0.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    }

    def __init__(self):
        self.log = []

    def track(self, model, input_tokens, output_tokens, purpose="general"):
        pricing = self.PRICING_PER_1M.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "purpose": purpose,
            "estimated_cost": cost,
        }
        self.log.append(entry)
        return entry

    def summary(self):
        total_in = sum(e["input_tokens"] for e in self.log)
        total_out = sum(e["output_tokens"] for e in self.log)
        total_cost = sum(e["estimated_cost"] for e in self.log)
        return {
            "total_calls": len(self.log),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "estimated_cost": f"${total_cost:.4f}",
        }
```

Wire it into your query endpoint. After the LLM call:

```python
tracker = CostTracker()

# After ollama.chat returns:
input_tokens = len(req.question) // 4   # rough estimate
output_tokens = len(answer) // 4
tracker.track("llama3.3:70b", input_tokens, output_tokens, purpose="rag_query")
```

Add a stats endpoint:

```python
@app.get("/stats")
async def get_stats():
    return {
        "cache": {
            "hits": cache.hits,
            "misses": cache.misses,
            "hit_rate": f"{cache.hits / max(cache.hits + cache.misses, 1):.1%}",
            "size": len(cache.cache),
        },
        "tokens": tracker.summary(),
    }
```

Restart, run a few queries, then check stats:

```bash
curl http://localhost:8000/stats | python3 -m json.tool
```

You'll see your cache hit rate, total token volume, and estimated cost. Right now cost is $0.00 because you're running local. But those token counts are real, and they translate directly to dollars when you switch to a cloud model.

---

## Step 7: Error Handling -- When Ollama Goes Down

What happens right now if Ollama crashes mid-query? Your users get a 500 Internal Server Error with a cryptic traceback. Not great.

Let's handle this gracefully:

```python
@app.post("/query")
async def query_kb(req: QueryRequest, request: Request):
    # ... rate limit check ...
    # ... cache check ...

    try:
        results = kb.query(query_texts=[req.question], n_results=req.n_results)
        context = "\n".join(results["documents"][0])
        source_ids = results["ids"][0]
    except Exception as e:
        raise HTTPException(status_code=503, detail="Knowledge base unavailable")

    try:
        response = llm.chat.completions.create(
            model="llama3.3:70b",
            messages=[
                {"role": "system", "content": "Answer using only the context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQ: {req.question}"},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        # Graceful degradation: return sources even if LLM fails
        return {
            "answer": "[LLM unavailable] Here are the relevant source documents.",
            "sources": source_ids,
            "raw_context": context,
            "cached": False,
            "degraded": True,
        }

    # ... rest of the endpoint ...
```

Notice the graceful degradation. If the LLM is down, we still return the retrieved documents. The user gets relevant reference material even without the generated answer. That's a lot better than a blank error page.

Think of it like this: if the automated inspection camera goes down on a production line, you don't stop the line entirely. You fall back to manual inspection. Same idea.

Test it by stopping Ollama and sending a query:

```bash
killall ollama
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the torque spec for M10 bolts?"}' | python3 -m json.tool
```

You should get a response with `"degraded": true` and the raw source documents. Start Ollama back up when you're done.

---

## Step 8: Request Logging Middleware

Let's add timing to every request. This is how you know if your API is getting slower over time:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -- {elapsed:.0f}ms")
    return response
```

Now every request gets logged with its latency. You'll see lines like:

```
INFO:rag-api:POST /query -- 2341ms
INFO:rag-api:POST /query -- 4ms       <-- cache hit!
INFO:rag-api:GET /health -- 12ms
```

That 2341ms vs 4ms difference? That's your cache working. And if you start seeing queries taking 10+ seconds, you know something is wrong before users complain.

---

## Exercise: The Full Production Server

Now let's put it all together. Here's the complete server with every production feature we've built:

```python
# 18-production-and-deployment/production_server.py
"""Production RAG API server with caching, rate limiting, health checks, and cost tracking."""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from openai import OpenAI
import chromadb
import hashlib
import json
import time
import logging
from collections import defaultdict
from datetime import datetime
import uvicorn

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    n_results: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    cached: bool
    latency_ms: float
    degraded: bool = False

# --- Cache ---
class ResponseCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, query, **kwargs):
        content = json.dumps({"query": query, **kwargs}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query, **kwargs):
        key = self._key(query, **kwargs)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, query, response, **kwargs):
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[self._key(query, **kwargs)] = response

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, max_requests=60, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def check(self, client_id):
        now = time.time()
        self.requests[client_id] = [
            t for t in self.requests[client_id] if now - t < self.window
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True

# --- Cost Tracker ---
class CostTracker:
    PRICING_PER_1M = {
        "llama3.3:70b": {"input": 0.00, "output": 0.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    }

    def __init__(self):
        self.log = []

    def track(self, model, input_tokens, output_tokens, purpose="general"):
        pricing = self.PRICING_PER_1M.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.log.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "purpose": purpose,
            "estimated_cost": cost,
        })

    def summary(self):
        total_in = sum(e["input_tokens"] for e in self.log)
        total_out = sum(e["output_tokens"] for e in self.log)
        total_cost = sum(e["estimated_cost"] for e in self.log)
        return {
            "total_calls": len(self.log),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "estimated_cost": f"${total_cost:.4f}",
        }

# --- Knowledge Base ---
def setup_kb():
    client = chromadb.Client()
    collection = client.create_collection(name="production_kb")
    docs = [
        ("MT-302", "Torque Spec MT-302: Frame #4200. M8=25-30Nm, M10=45-55Nm, M12=80-100Nm."),
        ("WPS-201", "WPS-201: GMAW carbon steel. ER70S-6. 75/25 Ar/CO2. Interpass 400F max."),
        ("QC-107", "Form QC-107: Visual inspection. Surface, welds, hardware, coating. Fail = HOLD tag."),
        ("SAFE-001", "LOTO SOP-SAFE-001: Notify, shutdown, isolate, lock/tag, release energy, verify."),
        ("PPE-001", "PPE: Glasses always. Welding: helmet shade 10-13, gloves, FR. Grinding: face shield."),
    ]
    collection.add(
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
    )
    return collection

# --- Initialize everything ---
llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
app = FastAPI(title="Manufacturing RAG API", version="1.0.0")
kb = setup_kb()
cache = ResponseCache()
rate_limiter = RateLimiter(max_requests=60, window_seconds=60)
tracker = CostTracker()

# --- Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -- {elapsed:.0f}ms")
    return response

# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def query_kb_endpoint(req: QueryRequest, request: Request):
    """Query the manufacturing knowledge base with RAG."""
    start = time.time()

    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Cache check
    cached = cache.get(req.question, n=req.n_results)
    if cached:
        return QueryResponse(**cached, cached=True,
                           latency_ms=(time.time() - start) * 1000)

    # Retrieve
    try:
        results = kb.query(query_texts=[req.question], n_results=req.n_results)
        context = "\n".join(results["documents"][0])
        source_ids = results["ids"][0]
    except Exception:
        raise HTTPException(status_code=503, detail="Knowledge base unavailable")

    # Generate
    try:
        response = llm.chat.completions.create(
            model="llama3.3:70b",
            messages=[
                {"role": "system", "content": "Answer using only the context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQ: {req.question}"},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content

        # Track tokens
        input_tokens = len(req.question) // 4
        output_tokens = len(answer) // 4
        tracker.track("llama3.3:70b", input_tokens, output_tokens, "rag_query")

    except Exception:
        # Graceful degradation
        return QueryResponse(
            answer="[LLM unavailable] See source documents.",
            sources=source_ids,
            cached=False,
            latency_ms=(time.time() - start) * 1000,
            degraded=True,
        )

    result = {"answer": answer, "sources": source_ids}
    cache.set(req.question, result, n=req.n_results)

    return QueryResponse(
        **result, cached=False, degraded=False,
        latency_ms=(time.time() - start) * 1000,
    )

@app.get("/health")
async def health_check():
    """Health check -- verifies all dependencies are reachable."""
    checks = {}
    try:
        llm.models.list()
        checks["ollama"] = "ok"
    except Exception as e:
        checks["ollama"] = f"error: {e}"

    try:
        checks["knowledge_base"] = f"ok ({kb.count()} docs)"
    except Exception as e:
        checks["knowledge_base"] = f"error: {e}"

    all_ok = all(str(v).startswith("ok") for v in checks.values())
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}

@app.get("/stats")
async def get_stats():
    """Cache, rate limiter, and token usage statistics."""
    return {
        "cache": {
            "hits": cache.hits,
            "misses": cache.misses,
            "hit_rate": f"{cache.hits / max(cache.hits + cache.misses, 1):.1%}",
            "size": len(cache.cache),
        },
        "tokens": tracker.summary(),
    }

if __name__ == "__main__":
    print("Starting Manufacturing RAG API...")
    print("  API docs: http://localhost:8000/docs")
    print("  Health:   http://localhost:8000/health")
    print("  Stats:    http://localhost:8000/stats")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run it:

```bash
python 18-production-and-deployment/production_server.py
```

Now put it through its paces from a second terminal:

```bash
# 1. Health check
curl http://localhost:8000/health | python3 -m json.tool

# 2. First query (cache miss)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What PPE do I need for welding?"}'

# 3. Same query (cache hit -- notice the speed)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What PPE do I need for welding?"}'

# 4. Different query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the torque spec for M10 bolts on Frame 4200?"}'

# 5. Check your stats
curl http://localhost:8000/stats | python3 -m json.tool

# 6. Try bad input (validation catches it)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Hi"}'

# 7. Browse the auto-generated docs
open http://localhost:8000/docs
```

Look at your stats output. You should see 1 cache hit, 2 cache misses, and token usage from the two LLM calls. The cached query should have come back in under 5ms vs. seconds for the fresh ones.

---

## What You Just Built

Let's recap what's running in that single Python file:

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| FastAPI endpoints | Typed HTTP API with auto-docs | Users interact via HTTP, not Python scripts |
| Response cache | Skips LLM for repeated queries | 4ms vs 2000ms for common questions |
| Rate limiter | Caps requests per client per minute | Prevents abuse and runaway costs |
| Health check | Verifies Ollama + KB are reachable | Monitoring knows when you're degraded |
| Cost tracker | Counts tokens per request | Know your volume before going to cloud |
| Graceful degradation | Returns sources even if LLM is down | Partial answer beats no answer |
| Request logging | Times every request | Spot performance regressions early |
| Input validation | Rejects malformed requests | No empty strings, no crazy long inputs |

---

## Takeaways

1. **FastAPI gives you a production API skeleton in minutes** -- typed models, validation, auto-docs, async support
2. **Cache aggressively** -- identical queries should never hit the LLM twice
3. **Rate limit from day one** -- it's trivial to add and prevents a whole class of problems
4. **Health checks should check dependencies** -- "server is up" is not the same as "server is working"
5. **Track tokens even when cost is zero** -- you're building the habit and the infrastructure for cloud migration
6. **Graceful degradation is better than hard failure** -- return what you can, even if it's incomplete

## What's Next

You have every piece. Module 19 is the capstone: you'll combine RAG, evaluation, guardrails, and these production patterns into one complete manufacturing task description system. Everything you've built across 18 modules, working together.
