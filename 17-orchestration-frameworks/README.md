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

All with plain Python, the OpenAI SDK pointed at Ollama, and ChromaDB. No frameworks.

Now let's see what happens when you use one.

---

## Step 1: Install the Frameworks

You'll need both LangChain and LlamaIndex. Let's install them:

```bash
pip install langchain-openai langchain-chroma langchain-core
```

```bash
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai
```

Let's verify they installed. Quick check:

```python
python3 -c "from langchain_openai import ChatOpenAI; print('LangChain ready')"
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

from openai import OpenAI
import chromadb

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Set up ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("plain_scouting_docs")

# Your football scouting documents
docs = [
    {"id": "qb101", "text": "QB-101: Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Arm strength: 62 mph. Weakness: locks onto first read under pressure."},
    {"id": "rb201", "text": "RB-201: Explosive runner with 4.38 40-yard dash. Exceptional vision, finds cutback lanes. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection."},
    {"id": "wr301", "text": "WR-301: Crisp route runner with elite separation. Full route tree, slot and outside. 4.42 speed, 38-inch vertical. 2.1% drop rate. Weakness: press coverage at the line."},
    {"id": "ol401", "text": "OL-401: Excellent pass protection anchor. Quick lateral movement. 34-inch arms. Run blocking: 82.5/100. 2 sacks allowed in 580 snaps. Weakness: combo blocks."},
    {"id": "def501", "text": "DEF-501: Cover-3 base with single-high safety. Press corners. Pattern-match zone on 3rd down. Aggressive nickel blitz. Weakness: crossing routes against zone."},
]

# Embed and store
for doc in docs:
    embedding = llm.embeddings.create(model="nomic-embed-text", input=doc["text"]).data[0].embedding
    collection.add(ids=[doc["id"]], documents=[doc["text"]], embeddings=[embedding])

# Query
question = "What's the 40 time and yards after contact for the RB prospect?"
q_embedding = llm.embeddings.create(model="nomic-embed-text", input=question).data[0].embedding
results = collection.query(query_embeddings=[q_embedding], n_results=2)

# Build prompt with context
context = "\n\n".join(results["documents"][0])
prompt = f"""Answer using ONLY this context. Cite sources.

Context:
{context}

Question: {question}

Answer:"""

response = llm.chat.completions.create(
    model="llama3.3:70b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)

print("=== Plain Code RAG ===")
print(f"Q: {question}")
print(f"A: {response.choices[0].message.content}")
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

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
```

That's a lot of imports. But watch what happens next:

```python
# Set up model and embeddings -- one line each
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="llama3.3:70b",
    temperature=0.0,
)
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="nomic-embed-text",
)
```

Create the vector store and load documents:

```python
docs = [
    Document(page_content="QB-101: Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Arm strength: 62 mph. Weakness: locks onto first read under pressure.",
             metadata={"source": "QB-101"}),
    Document(page_content="RB-201: Explosive runner with 4.38 40-yard dash. Exceptional vision, finds cutback lanes. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection.",
             metadata={"source": "RB-201"}),
    Document(page_content="WR-301: Crisp route runner with elite separation. Full route tree, slot and outside. 4.42 speed, 38-inch vertical. 2.1% drop rate. Weakness: press coverage at the line.",
             metadata={"source": "WR-301"}),
    Document(page_content="OL-401: Excellent pass protection anchor. Quick lateral movement. 34-inch arms. Run blocking: 82.5/100. 2 sacks allowed in 580 snaps. Weakness: combo blocks.",
             metadata={"source": "OL-401"}),
    Document(page_content="DEF-501: Cover-3 base with single-high safety. Press corners. Pattern-match zone on 3rd down. Aggressive nickel blitz. Weakness: crossing routes against zone.",
             metadata={"source": "DEF-501"}),
]

vectorstore = Chroma.from_documents(docs, embeddings, collection_name="lc_scouting_docs")
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
question = "What's the 40 time and yards after contact for the RB prospect?"
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
# Instead of (local Ollama):
# llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", model="llama3.3:70b")

# Switch to OpenAI cloud -- just change the constructor:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # uses OPENAI_API_KEY env var
```

Everything else stays the same. The chain, the prompt, the retriever -- none of it changes.

Want to swap ChromaDB for Pinecone? Same idea:

```python
# Instead of Chroma, use Pinecone:
# from langchain_pinecone import PineconeVectorStore
# vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name="scouting")
```

That's the value proposition of frameworks: swappable components. For a team project where you might switch models or vector stores, that's huge.

---

## Step 5: The Same Pipeline in LlamaIndex

Now let's try LlamaIndex. It takes a different approach -- laser focused on documents and retrieval:

```python
# 17-orchestration-frameworks/ex3_llamaindex_rag.py
"""The same RAG pipeline, built with LlamaIndex."""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure globally -- LlamaIndex uses a Settings object
Settings.llm = LlamaOpenAI(
    model="llama3.3:70b",
    temperature=0.0,
    api_base="http://localhost:11434/v1",
    api_key="ollama",
)
Settings.embed_model = OpenAIEmbedding(
    model_name="nomic-embed-text",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
)
```

Now create documents and build the index:

```python
documents = [
    Document(text="QB-101: Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Arm strength: 62 mph. Weakness: locks onto first read under pressure.",
             metadata={"source": "QB-101"}),
    Document(text="RB-201: Explosive runner with 4.38 40-yard dash. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection.",
             metadata={"source": "RB-201"}),
    Document(text="WR-301: Crisp route runner with elite separation. Full route tree. 4.42 speed, 38-inch vertical. 2.1% drop rate. Weakness: press coverage.",
             metadata={"source": "WR-301"}),
    Document(text="DEF-501: Cover-3 base with single-high safety. Press corners. Pattern-match zone on 3rd down. Aggressive nickel blitz. Weakness: crossing routes against zone.",
             metadata={"source": "DEF-501"}),
]

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)
```

That's it. Two lines to go from documents to a working query engine. LlamaIndex handled the chunking, embedding, and indexing. Now query it:

```python
question = "What's the 40 time for the RB prospect?"
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
    llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    # call llm.chat.completions.create() and llm.embeddings.create() directly

LangChain:
    llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", model="llama3.3:70b")
    embeddings = OpenAIEmbeddings(base_url="http://localhost:11434/v1", api_key="ollama", model="nomic-embed-text")

LlamaIndex:
    Settings.llm = LlamaOpenAI(model="llama3.3:70b", api_base="http://localhost:11434/v1", api_key="ollama")
    Settings.embed_model = OpenAIEmbedding(model_name="nomic-embed-text", api_base="http://localhost:11434/v1", api_key="ollama")


EMBEDDING + STORING DOCUMENTS
==============================

Plain code:
    for doc in docs:
        emb = llm.embeddings.create(model="nomic-embed-text", input=doc["text"]).data[0].embedding
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
    q_emb = llm.embeddings.create(model="nomic-embed-text", input=question).data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=2)
    context = "\\n".join(results["documents"][0])
    response = llm.chat.completions.create(model="llama3.3:70b", messages=[...])
    answer = response.choices[0].message.content

LangChain:
    answer = rag_chain.invoke(question)

LlamaIndex:
    response = query_engine.query(question)


SWAPPING TO CLOUD (e.g., OpenAI GPT-4)
==============================

Plain code:
    # Remove base_url and api_key="ollama", set OPENAI_API_KEY env var
    llm = OpenAI()  # uses OPENAI_API_KEY from environment

LangChain:
    # Change: llm = ChatOpenAI(base_url="...", api_key="ollama", model="llama3.3:70b")
    #      to: llm = ChatOpenAI(model="gpt-4o-mini")  # uses OPENAI_API_KEY
    # Everything else stays the same.

LlamaIndex:
    # Change: Settings.llm = LlamaOpenAI(api_base="...", api_key="ollama", model="llama3.3:70b")
    #      to: Settings.llm = LlamaOpenAI(model="gpt-4o-mini")  # uses OPENAI_API_KEY
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
- Your project is document-heavy -- lots of PDFs, game film notes, scouting reports to index
- You want automatic chunking strategies (sentence, paragraph, semantic)
- Citation tracking is important (which report answered which question)
- You're building a pure RAG system without complex agent logic

**For your football scouting capstone:**
- Your golden dataset + evaluation pipeline = plain code is fine
- If you add dozens of scouting reports and film notes = LlamaIndex shines
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
     "label": "Solo dev, simple RAG, 10 scouting reports"},

    {"num_developers": 3, "may_switch_models": True, "num_documents": 200,
     "has_agent_logic": False, "pipeline_steps": 3,
     "label": "Team of 3, 200 scouting reports, might switch to cloud LLM"},

    {"num_developers": 2, "may_switch_models": True, "num_documents": 50,
     "has_agent_logic": True, "pipeline_steps": 5,
     "label": "2 devs, draft analyst agent with tools, 50 reports, complex pipeline"},

    {"num_developers": 1, "may_switch_models": False, "num_documents": 100,
     "has_agent_logic": True, "pipeline_steps": 4,
     "label": "Solo dev, agent + lots of scouting reports"},
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
- Lots of scouting reports? LlamaIndex.
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
2. **LlamaIndex** = document-focused, great for RAG-heavy applications with lots of scouting reports to index
3. **Plain code is often the right choice** -- add a framework when complexity demands it, not before
4. **Frameworks help teams more than solo devs** -- shared abstractions reduce onboarding time
5. **Evaluation matters more than framework choice** -- a well-tested plain-code pipeline beats an untested LangChain one every time

## Next Up

You can build it. But can you ship it? Module 18 covers production and deployment -- API servers, caching, error handling, and the patterns that turn a prototype into a system people rely on.
