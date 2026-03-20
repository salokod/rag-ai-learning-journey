# Module 17: Orchestration Frameworks

## Goal
See what LangChain and LlamaIndex offer, compare them to your plain code, and know when each makes sense.

---

## You Already Built Everything From Scratch

Over the last 16 modules, you built:
- Embeddings and vector search (Module 05)
- A full RAG pipeline (Module 06)
- Agents with tool calling (Module 15)
- Guardrails and safety checks (Module 16)

All with plain Python, Ollama, and ChromaDB. No frameworks.

Now let's see what happens when you use one.

---

## Step 1: Install the Frameworks

You'll need both LangChain and LlamaIndex. Let's install them:

```bash
pip install langchain-ollama langchain-chroma langchain-core
```

```bash
pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-ollama
```

Let's verify they installed. Quick check:

```python
python3 -c "from langchain_ollama import ChatOllama; print('LangChain ready')"
```

```python
python3 -c "from llama_index.core import VectorStoreIndex; print('LlamaIndex ready')"
```

Both should print their "ready" message. If either fails, check the pip install output for errors.

---

## Step 2: Your Plain-Code RAG Pipeline (The Baseline)

First, let's remind ourselves what your hand-built RAG looks like. This is roughly what you built in Module 06:

```python
# 17-orchestration-frameworks/ex1_plain_rag.py
"""Your plain-code RAG pipeline -- the baseline to compare against."""

import ollama
import chromadb

# Set up ChromaDB
client = chromadb.Client()
collection = client.create_collection("plain_mfg_docs")

# Your manufacturing documents
docs = [
    {"id": "mt302", "text": "Torque Spec MT-302: Frame #4200. M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern. Form QC-110."},
    {"id": "wps201", "text": "WPS-201: GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH. Interpass max 400F."},
    {"id": "qc107", "text": "Form QC-107: Visual inspection checklist. Check surface finish, weld quality, hardware. All items must pass."},
    {"id": "safety", "text": "LOTO SOP-SAFE-001: Notify operators, normal shutdown, isolate energy, lock/tag, release stored energy, verify zero."},
    {"id": "ppe", "text": "PPE: Safety glasses always. Welding: auto-dark helmet shade 10-13, leather gloves, FR clothing."},
]

# Embed and store
for doc in docs:
    embedding = ollama.embed(model="nomic-embed-text", input=doc["text"])["embeddings"][0]
    collection.add(ids=[doc["id"]], documents=[doc["text"]], embeddings=[embedding])

# Query
question = "What is the torque spec for M10 bolts on Frame #4200?"
q_embedding = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
results = collection.query(query_embeddings=[q_embedding], n_results=2)

# Build prompt with context
context = "\n\n".join(results["documents"][0])
prompt = f"""Answer using ONLY this context. Cite sources.

Context:
{context}

Question: {question}

Answer:"""

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.0},
)

print("=== Plain Code RAG ===")
print(f"Q: {question}")
print(f"A: {response['message']['content']}")
```

Run it:

```bash
python3 17-orchestration-frameworks/ex1_plain_rag.py
```

Count the lines. That's roughly 30 lines of working code (not counting the doc data). It works. You understand every line. Let's see how the frameworks compare.

---

## Step 3: The Same Pipeline in LangChain

```python
# 17-orchestration-frameworks/ex2_langchain_rag.py
"""The same RAG pipeline, built with LangChain."""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
```

That's a lot of imports. But watch what happens next:

```python
# Set up model and embeddings -- one line each
llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

Create the vector store and load documents:

```python
docs = [
    Document(page_content="Torque Spec MT-302: Frame #4200. M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern. Form QC-110.",
             metadata={"source": "MT-302"}),
    Document(page_content="WPS-201: GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH. Interpass max 400F.",
             metadata={"source": "WPS-201"}),
    Document(page_content="Form QC-107: Visual inspection checklist. Check surface finish, weld quality, hardware. All items must pass.",
             metadata={"source": "QC-107"}),
    Document(page_content="LOTO SOP-SAFE-001: Notify operators, normal shutdown, isolate energy, lock/tag, release stored energy, verify zero.",
             metadata={"source": "SOP-SAFE-001"}),
    Document(page_content="PPE: Safety glasses always. Welding: auto-dark helmet shade 10-13, leather gloves, FR clothing.",
             metadata={"source": "PPE-001"}),
]

vectorstore = Chroma.from_documents(docs, embeddings, collection_name="lc_mfg_docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

Notice: no manual embedding calls. `Chroma.from_documents` handled that. Now build the chain:

```python
prompt = ChatPromptTemplate.from_template("""Answer using ONLY the context below. Cite sources.

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']}]: {d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

That chain reads like a sentence: retrieve docs, format them, fill the prompt, send to LLM, parse the output. Now use it:

```python
question = "What is the torque spec for M10 bolts on Frame #4200?"
print("=== LangChain RAG ===")
print(f"Q: {question}")
print(f"A: {rag_chain.invoke(question)}")
```

Run it:

```bash
python3 17-orchestration-frameworks/ex2_langchain_rag.py
```

Same answer. Fewer lines of "plumbing" code. But here's the real payoff -- watch what happens when you want to change something.

---

## Step 4: The LangChain Payoff -- Swap One Line

Want to switch from Ollama to OpenAI? In your plain code, you'd rewrite the embedding calls, the chat calls, and the response parsing.

In LangChain, you change ONE line:

```python
# Instead of:
# llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

# You'd write:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
```

Everything else stays the same. The chain, the prompt, the retriever -- none of it changes.

Want to swap ChromaDB for Pinecone? Same idea:

```python
# Instead of Chroma, use Pinecone:
# from langchain_pinecone import PineconeVectorStore
# vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name="mfg")
```

That's the value proposition of frameworks: swappable components. For a team project where you might switch models or vector stores, that's huge.

---

## Step 5: The Same Pipeline in LlamaIndex

Now let's try LlamaIndex. It takes a different approach -- laser focused on documents and retrieval:

```python
# 17-orchestration-frameworks/ex3_llamaindex_rag.py
"""The same RAG pipeline, built with LlamaIndex."""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configure globally -- LlamaIndex uses a Settings object
Settings.llm = Ollama(model="llama3.1:8b", temperature=0.0, request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
```

Now create documents and build the index:

```python
documents = [
    Document(text="Torque Spec MT-302: Frame #4200. M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern.",
             metadata={"source": "MT-302"}),
    Document(text="WPS-201: GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH.",
             metadata={"source": "WPS-201"}),
    Document(text="Form QC-107: Visual inspection checklist. All items must pass. Fail = red HOLD tag.",
             metadata={"source": "QC-107"}),
    Document(text="LOTO SOP-SAFE-001: Notify, shutdown, isolate, lock/tag, release energy, verify zero.",
             metadata={"source": "SOP-SAFE-001"}),
]

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)
```

That's it. Two lines to go from documents to a working query engine. LlamaIndex handled the chunking, embedding, and indexing. Now query it:

```python
question = "What is the torque spec for M10 bolts?"
print("=== LlamaIndex RAG ===")
print(f"Q: {question}")
response = query_engine.query(question)
print(f"A: {response}")
print(f"Sources: {[n.metadata.get('source', '?') for n in response.source_nodes]}")
```

Run it:

```bash
python3 17-orchestration-frameworks/ex3_llamaindex_rag.py
```

Same answer again. Even less code than LangChain for the RAG-specific parts. And notice how it automatically tracks source nodes -- you get citations for free.

---

## Step 6: Side-by-Side Comparison

Let's put all three approaches next to each other. Create this comparison file:

```python
# 17-orchestration-frameworks/ex4_comparison.py
"""Side-by-side: plain code vs LangChain vs LlamaIndex."""

print("""
SETTING UP MODEL + EMBEDDINGS
==============================

Plain code:
    # No setup -- just call ollama.chat() and ollama.embed() directly

LangChain:
    llm = ChatOllama(model="llama3.1:8b")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

LlamaIndex:
    Settings.llm = Ollama(model="llama3.1:8b")
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


EMBEDDING + STORING DOCUMENTS
==============================

Plain code:
    for doc in docs:
        emb = ollama.embed(model="nomic-embed-text", input=doc["text"])["embeddings"][0]
        collection.add(ids=[doc["id"]], documents=[doc["text"]], embeddings=[emb])

LangChain:
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

LlamaIndex:
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)


QUERYING
==============================

Plain code:
    q_emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
    results = collection.query(query_embeddings=[q_emb], n_results=2)
    context = "\\n".join(results["documents"][0])
    response = ollama.chat(model="llama3.1:8b", messages=[...])
    answer = response["message"]["content"]

LangChain:
    answer = rag_chain.invoke(question)

LlamaIndex:
    response = query_engine.query(question)


SWAPPING THE MODEL
==============================

Plain code:
    # Change every ollama.chat() call. Update response parsing.
    # If switching providers, rewrite API calls entirely.

LangChain:
    # Change: llm = ChatOllama(...) -> llm = ChatOpenAI(...)
    # Everything else stays the same.

LlamaIndex:
    # Change: Settings.llm = Ollama(...) -> Settings.llm = OpenAI(...)
    # Everything else stays the same.


LINE COUNT (approximate, excluding doc data)
==============================

Plain code:    ~30 lines
LangChain:     ~15 lines
LlamaIndex:    ~10 lines
""")
```

Run it:

```bash
python3 17-orchestration-frameworks/ex4_comparison.py
```

Read through the comparison. The patterns are clear:
- Plain code: more lines, full control, you see everything
- LangChain: fewer lines, lots of imports, very swappable
- LlamaIndex: fewest lines for RAG, automatic chunking/indexing

---

## Step 7: The Honest Assessment

Let's be real about when each approach makes sense.

**Use plain code when:**
- You're learning (you already did this -- now you understand what the frameworks hide)
- Your pipeline is simple and stable -- embed, retrieve, generate, done
- Performance matters -- frameworks add overhead
- You want to debug easily -- no framework magic to trace through
- You're the only developer

**Use LangChain when:**
- You need complex chains: retrieve -> check -> generate -> validate -> maybe retry
- You might swap models or vector stores (Ollama today, OpenAI tomorrow)
- Multiple developers need to understand the codebase
- You want the agent/tool-calling framework (LangChain's agent support is mature)
- You need streaming, batching, or async out of the box

**Use LlamaIndex when:**
- Your project is document-heavy -- lots of PDFs, manuals, specs to index
- You want automatic chunking strategies (sentence, paragraph, semantic)
- Citation tracking is important (which document answered which question)
- You're building a pure RAG system without complex agent logic

**For your manufacturing capstone:**
- Your golden dataset + evaluation pipeline = plain code is fine
- If you add dozens of manufacturing documents = LlamaIndex shines
- If you build a multi-step agent with tools + guardrails + RAG = LangChain helps

Here's the thing most framework tutorials won't tell you: **your plain-code pipeline from Modules 05-08 is perfectly fine for production.** Frameworks help when complexity grows. Don't adopt one just because it's popular.

---

## Step 8: A Decision Framework

When someone asks "should I use a framework?", walk through this:

```python
# 17-orchestration-frameworks/ex5_decision.py
"""A practical decision framework for choosing your approach."""

def choose_approach(
    num_developers: int,
    may_switch_models: bool,
    num_documents: int,
    has_agent_logic: bool,
    pipeline_steps: int,
) -> str:
    """Suggest an approach based on project characteristics."""

    reasons = []

    # Solo dev with simple pipeline? Plain code.
    if num_developers == 1 and pipeline_steps <= 3 and not may_switch_models:
        reasons.append("Solo dev, simple pipeline -> plain code")
        return f"RECOMMENDATION: Plain code. {'; '.join(reasons)}"

    # Lots of documents? LlamaIndex for the RAG layer.
    if num_documents > 50:
        reasons.append(f"{num_documents} documents -> LlamaIndex for indexing/retrieval")

    # Complex agent logic? LangChain.
    if has_agent_logic and pipeline_steps > 3:
        reasons.append(f"{pipeline_steps}-step pipeline with agents -> LangChain for orchestration")

    # Team project with model flexibility? Framework.
    if num_developers > 1:
        reasons.append(f"{num_developers} developers -> framework for shared abstractions")
    if may_switch_models:
        reasons.append("May switch models -> framework for swappability")

    if not reasons:
        return "RECOMMENDATION: Plain code. Simple enough to not need a framework."

    # If both document-heavy and agent-heavy, can combine
    has_llamaindex = any("LlamaIndex" in r for r in reasons)
    has_langchain = any("LangChain" in r for r in reasons)

    if has_llamaindex and has_langchain:
        return f"RECOMMENDATION: LlamaIndex for RAG + LangChain for agents. {'; '.join(reasons)}"
    elif has_llamaindex:
        return f"RECOMMENDATION: LlamaIndex. {'; '.join(reasons)}"
    elif has_langchain:
        return f"RECOMMENDATION: LangChain. {'; '.join(reasons)}"
    else:
        return f"RECOMMENDATION: LangChain (general purpose). {'; '.join(reasons)}"


# Try different scenarios
scenarios = [
    {"num_developers": 1, "may_switch_models": False, "num_documents": 10,
     "has_agent_logic": False, "pipeline_steps": 2,
     "label": "Solo dev, simple RAG, 10 docs"},

    {"num_developers": 3, "may_switch_models": True, "num_documents": 200,
     "has_agent_logic": False, "pipeline_steps": 3,
     "label": "Team of 3, 200 manufacturing docs, might switch to cloud LLM"},

    {"num_developers": 2, "may_switch_models": True, "num_documents": 50,
     "has_agent_logic": True, "pipeline_steps": 5,
     "label": "2 devs, agent with tools, 50 docs, complex pipeline"},

    {"num_developers": 1, "may_switch_models": False, "num_documents": 100,
     "has_agent_logic": True, "pipeline_steps": 4,
     "label": "Solo dev, agent + lots of docs"},
]

print("=== Framework Decision Tool ===\n")
for s in scenarios:
    label = s.pop("label")
    print(f"Scenario: {label}")
    print(f"  {choose_approach(**s)}\n")
```

Run it:

```bash
python3 17-orchestration-frameworks/ex5_decision.py
```

The recommendations match your intuition:
- Simple solo project? Plain code.
- Lots of docs? LlamaIndex.
- Complex agent pipeline with a team? LangChain.
- Both? Combine them.

---

## What You Learned

1. **LangChain** reduces plumbing code and makes components swappable -- change one line to switch models or vector stores
2. **LlamaIndex** is even more concise for RAG -- automatic chunking, embedding, and source tracking
3. **Plain code** gives you full control and is perfectly fine for simple, stable pipelines
4. **Frameworks help teams** -- shared abstractions mean everyone speaks the same language
5. **The right choice depends on your project** -- not on what's trending on GitHub

The most important thing: you built everything from scratch first. Now you UNDERSTAND what these frameworks are doing under the hood. When LangChain's `rag_chain.invoke()` runs, you know it's embedding the query, searching the vector store, building a prompt, and calling the LLM. No magic -- just the same steps you wrote by hand.

---

## Takeaways

1. **LangChain** = general-purpose orchestration, great for agents, chains, and swappable components
2. **LlamaIndex** = document-focused, great for RAG-heavy applications with lots of files to index
3. **Plain code is often the right choice** -- add a framework when complexity demands it, not before
4. **Frameworks help teams more than solo devs** -- shared abstractions reduce onboarding time
5. **Evaluation matters more than framework choice** -- a well-tested plain-code pipeline beats an untested LangChain one every time

## Next Up

You can build it. But can you ship it? Module 18 covers production and deployment -- API servers, caching, error handling, and the patterns that turn a prototype into a system people rely on.
