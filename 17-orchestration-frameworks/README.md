# Module 17: Orchestration Frameworks

## Goal
Learn LangChain and LlamaIndex — the two dominant frameworks for building LLM applications. Understand when to use each, and when to use neither (plain code).

---

## Concepts

### Why Orchestration Frameworks?

So far, you've built everything with raw Ollama calls and custom code. That works great for learning. But in production:
- You need standard patterns for RAG pipelines
- You want plug-and-play components (swap vector stores, models, etc.)
- You need reliable chains of LLM calls
- Teams need common abstractions

### LangChain vs. LlamaIndex

| | LangChain | LlamaIndex |
|---|---|---|
| **Focus** | General LLM orchestration | Data-centric RAG pipelines |
| **Strength** | Flexible chains, agents, tools | Document indexing & retrieval |
| **Best for** | Multi-step workflows, agents | RAG-heavy applications |
| **Learning curve** | Medium-high (lots of abstractions) | Medium (more focused) |
| **Your use case** | Agent that generates + evaluates | Indexing manufacturing docs |

### When to Use Plain Code Instead

Frameworks add complexity. Use them when:
- ✓ You need to swap components frequently (models, vector stores)
- ✓ Multiple team members need to understand the codebase
- ✓ You're building complex multi-step pipelines

Use plain code when:
- ✓ Simple, single-purpose pipeline
- ✓ Performance is critical (frameworks add overhead)
- ✓ You want full control and debuggability

---

## Exercise 1: RAG with LangChain

```python
# 17-orchestration-frameworks/ex1_langchain_rag.py
"""Build your manufacturing RAG pipeline using LangChain."""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# === Step 1: Set up components ===
llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# === Step 2: Create and populate vector store ===
docs = [
    Document(
        page_content="Torque Spec MT-302: Frame #4200. M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern. Form QC-110.",
        metadata={"source": "MT-302", "type": "specification"},
    ),
    Document(
        page_content="WPS-201: GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH. Interpass max 400°F. Visual + UT per AWS D1.1.",
        metadata={"source": "WPS-201", "type": "specification"},
    ),
    Document(
        page_content="Form QC-107: Visual inspection. Check surface finish, weld quality, hardware, paint/coating. All items must pass. Fail = red HOLD tag.",
        metadata={"source": "QC-107", "type": "form"},
    ),
    Document(
        page_content="LOTO SOP-SAFE-001: Notify operators, normal shutdown, isolate energy, lock/tag, release stored energy, verify zero energy. Only lock owner removes.",
        metadata={"source": "SOP-SAFE-001", "type": "safety"},
    ),
    Document(
        page_content="PPE: Safety glasses always. Welding: auto-dark helmet shade 10-13, leather gloves, FR clothing. Grinding: face shield, hearing protection.",
        metadata={"source": "PPE-001", "type": "safety"},
    ),
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="langchain_manufacturing",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# === Step 3: Build the RAG chain ===
prompt = ChatPromptTemplate.from_template("""You are a manufacturing knowledge assistant.
Answer the question using ONLY the provided context. Cite source documents.
If the answer isn't in the context, say "Not found in available documents."

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

# === Step 4: Use it! ===
questions = [
    "What is the torque spec for M10 bolts on Frame #4200?",
    "What PPE do I need for welding?",
    "How do I perform lockout/tagout?",
]

print("=== LangChain RAG Pipeline ===\n")
for q in questions:
    print(f"Q: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}\n")

print("=== LangChain Benefits ===")
print("- retriever | format_docs | prompt | llm | parser — clean pipeline")
print("- Swap Ollama → OpenAI by changing one line")
print("- Swap Chroma → Pinecone by changing one line")
print("- Built-in streaming, batching, and async support")
```

---

## Exercise 2: RAG with LlamaIndex

```python
# 17-orchestration-frameworks/ex2_llamaindex_rag.py
"""Build the same pipeline using LlamaIndex — compare the approach."""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configure LlamaIndex to use local models
Settings.llm = Ollama(model="llama3.1:8b", temperature=0.0, request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Create documents
documents = [
    Document(text="Torque Spec MT-302: Frame #4200. M8=25-30Nm, M10=45-55Nm. Star pattern. Form QC-110.",
             metadata={"source": "MT-302"}),
    Document(text="WPS-201: GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2. Interpass max 400°F.",
             metadata={"source": "WPS-201"}),
    Document(text="Form QC-107: Visual inspection checklist. All items must pass. Fail = red HOLD tag.",
             metadata={"source": "QC-107"}),
    Document(text="LOTO SOP-SAFE-001: Notify, shutdown, isolate, lock/tag, release energy, verify zero.",
             metadata={"source": "SOP-SAFE-001"}),
]

# Build index (handles chunking + embedding automatically)
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=2)

# Query
questions = [
    "What is the torque spec for M10 bolts?",
    "What is the lockout procedure?",
]

print("=== LlamaIndex RAG Pipeline ===\n")
for q in questions:
    print(f"Q: {q}")
    response = query_engine.query(q)
    print(f"A: {response}\n")
    print(f"Sources: {[n.metadata.get('source', '?') for n in response.source_nodes]}\n")

print("=== LlamaIndex Benefits ===")
print("- Fewer lines of code for RAG")
print("- Automatic chunking and indexing")
print("- Built-in source node tracking")
print("- Great for document-heavy applications")
```

---

## Exercise 3: Framework Comparison

```python
# 17-orchestration-frameworks/ex3_comparison.py
"""Compare plain code, LangChain, and LlamaIndex approaches."""

print("""
╔══════════════════════════════════════════════════════════════════╗
║           FRAMEWORK COMPARISON FOR YOUR PROJECT                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  PLAIN CODE (Modules 05-08)                                    ║
║  + Full control, no dependencies, easy to debug                ║
║  + Best performance (no framework overhead)                    ║
║  - More code to write and maintain                             ║
║  - Have to implement patterns yourself                         ║
║  VERDICT: Great for learning and simple pipelines              ║
║                                                                ║
║  LANGCHAIN                                                     ║
║  + Most popular, huge ecosystem                                ║
║  + Great for agents, complex chains, tool use                  ║
║  + Easy to swap components (models, vector stores)             ║
║  - Lots of abstractions (can be confusing)                     ║
║  - Rapid API changes (docs often outdated)                     ║
║  VERDICT: Good for complex multi-step workflows                ║
║                                                                ║
║  LLAMAINDEX                                                    ║
║  + Best for document-heavy RAG                                 ║
║  + Automatic chunking, indexing, retrieval                     ║
║  + Cleaner API for RAG-specific tasks                          ║
║  - Narrower focus than LangChain                               ║
║  - Less flexible for non-RAG tasks                             ║
║  VERDICT: Great for your manufacturing doc RAG system          ║
║                                                                ║
║  RECOMMENDATION FOR YOUR CAPSTONE:                             ║
║  Use LangChain for the agent/tool-use parts and LlamaIndex    ║
║  for the document indexing — OR plain code for everything      ║
║  (you already know how). Framework choice matters less than    ║
║  evaluation quality.                                           ║
╚══════════════════════════════════════════════════════════════════╝
""")
```

---

## Takeaways

1. **LangChain** = general-purpose LLM orchestration, great for agents and complex chains
2. **LlamaIndex** = document-focused, great for RAG-heavy applications
3. **Plain code** is often the best starting point — add frameworks when complexity demands it
4. **Frameworks help teams** — shared abstractions make codebases more maintainable
5. **Evaluation matters more than framework choice** — a well-tested plain-code pipeline beats an untested LangChain one

## Setting the Stage for Module 18

You can build it. But can you **ship** it? Module 18 covers production concerns: API servers, caching, rate limiting, error handling, cost management, and deployment patterns for LLM applications.
