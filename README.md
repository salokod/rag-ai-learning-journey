# AI & RAG Learning Journey

A hands-on, comprehensive learning path from LLM fundamentals to production-grade AI systems with rigorous evaluation. Built for a MacBook M4 Pro (48GB RAM) — everything runs locally.

## Who This Is For

You're a developer or technical professional who needs to confidently build, evaluate, and deploy LLM-powered systems in a professional environment. You need to know not just how to get an LLM to generate content, but how to **prove** it's generating *good* content — and catch it when it isn't.

## The Core Philosophy

> "If you can't measure it, you can't improve it."

Every module teaches you a capability **and** how to evaluate that capability. Testing isn't a separate phase — it's woven into everything. By the end, you'll be able to walk into a meeting and say "the AI-generated task descriptions are 23% more consistent than the manually written ones, and here's the evaluation framework that proves it."

## Your Hardware

- **MacBook M4 Pro, 48GB RAM** — plenty for running local LLMs up to ~32B parameters
- **Ollama** as the primary local LLM runtime
- **Recommended models**: Qwen 3 32B (Q4), Llama 3.3 8B, Mistral 7B
- No GPU server needed until fine-tuning (Module 14), and even then we have local options

## Learning Path Overview

### Phase 1: Foundations (Modules 0-4)
*Build your mental model of how LLMs work and how to control them*

| Module | Topic | What You'll Build |
|--------|-------|-------------------|
| [00](./00-environment-setup/) | Environment Setup | Python env, Ollama, all tooling ready |
| [01](./01-llm-fundamentals/) | LLM Fundamentals | Tokenizer explorer, temperature experiments |
| [02](./02-running-local-llms/) | Running Local LLMs | Local inference server, model comparison tool |
| [03](./03-prompt-engineering/) | Prompt Engineering | Style-matching prompt suite for manufacturing docs |
| [04](./04-structured-output/) | Structured Output | JSON/schema-constrained output parser |

### Phase 2: RAG Deep Dive (Modules 5-8)
*Learn to give LLMs the right context at the right time*

| Module | Topic | What You'll Build |
|--------|-------|-------------------|
| [05](./05-embeddings-and-vectors/) | Embeddings & Vector Stores | Semantic search engine over documents |
| [06](./06-rag-fundamentals/) | RAG Fundamentals | End-to-end RAG pipeline |
| [07](./07-advanced-rag/) | Advanced RAG | Hybrid search, re-ranking, query transformation |
| [08](./08-document-processing/) | Document Processing | PDF/DOCX ingestion pipeline |

### Phase 3: Evaluation & Testing (Modules 9-13)
*THE critical skill — prove your system works and catch regressions*

| Module | Topic | What You'll Build |
|--------|-------|-------------------|
| [09](./09-evaluation-fundamentals/) | Evaluation Fundamentals | Custom scoring functions, LLM-as-judge |
| [10](./10-rag-evaluation-with-ragas/) | RAG Evaluation (Ragas) | Full RAG evaluation pipeline with metrics |
| [11](./11-testing-with-deepeval/) | Testing (DeepEval) | pytest-style LLM test suite with CI integration |
| [12](./12-observability-with-langfuse/) | Observability (Langfuse) | Tracing, monitoring, and debugging dashboard |
| [13](./13-evaluation-datasets-and-benchmarks/) | Eval Datasets & Benchmarks | Golden dataset builder, regression test suite |

> **Galileo Alternative**: [Langfuse](https://langfuse.com) is the closest open-source equivalent to Galileo. It's MIT-licensed, self-hostable, and covers tracing, prompt management, and evaluations. We use it extensively in Module 12, and it integrates with Ragas and DeepEval from Modules 10-11.

### Phase 4: Advanced Topics (Modules 14-18)
*Level up with fine-tuning, agents, safety, and production patterns*

| Module | Topic | What You'll Build |
|--------|-------|-------------------|
| [14](./14-fine-tuning/) | Fine-tuning | LoRA fine-tuned model on your own data |
| [15](./15-agents-and-tool-use/) | Agents & Tool Use | Multi-step agent with tool calling |
| [16](./16-guardrails-and-safety/) | Guardrails & Safety | Input/output validation pipeline |
| [17](./17-orchestration-frameworks/) | Orchestration Frameworks | LangChain & LlamaIndex comparison |
| [18](./18-production-and-deployment/) | Production & Deployment | API server with caching, rate limiting, monitoring |

### Phase 5: Capstone (Module 19)
*Bring it all together*

| Module | Topic | What You'll Build |
|--------|-------|-------------------|
| [19](./19-capstone-manufacturing-tasks/) | **Capstone: Manufacturing Task Descriptions** | Full production system: RAG + evaluation + style matching + monitoring |

## How Each Module Works

Every module follows the same structure:

1. **Concepts** — What you need to understand, explained plainly
2. **Environment** — Any new tools/packages for this module
3. **Exercises** — Short, working code examples (15-30 min each)
4. **Takeaways** — What you should now be able to do
5. **Setting the Stage** — What's coming next and why it matters

## Quick Start

```bash
# Clone the repo
git clone https://github.com/salokod/rag-ai-learning-journey.git
cd rag-ai-learning-journey

# Start with Module 00 for full setup instructions
cat 00-environment-setup/README.md
```

## Key Tools You'll Master

| Tool | What It Does | Modules |
|------|-------------|---------|
| **Ollama** | Run LLMs locally | 02+ |
| **ChromaDB** | Vector database for embeddings | 05+ |
| **Ragas** | RAG-specific evaluation metrics | 10, 13, 19 |
| **DeepEval** | pytest-style LLM testing | 11, 13, 19 |
| **Langfuse** | LLM observability & tracing (Galileo alternative) | 12+ |
| **LangChain** | LLM orchestration framework | 17, 19 |
| **LlamaIndex** | Data framework for LLM apps | 17 |
| **Hugging Face** | Model hub, fine-tuning tools | 14 |

## Progress Tracking

Check off modules as you complete them:

- [ ] 00 - Environment Setup
- [ ] 01 - LLM Fundamentals
- [ ] 02 - Running Local LLMs
- [ ] 03 - Prompt Engineering
- [ ] 04 - Structured Output
- [ ] 05 - Embeddings & Vector Stores
- [ ] 06 - RAG Fundamentals
- [ ] 07 - Advanced RAG
- [ ] 08 - Document Processing
- [ ] 09 - Evaluation Fundamentals
- [ ] 10 - RAG Evaluation with Ragas
- [ ] 11 - Testing with DeepEval
- [ ] 12 - Observability with Langfuse
- [ ] 13 - Evaluation Datasets & Benchmarks
- [ ] 14 - Fine-tuning
- [ ] 15 - Agents & Tool Use
- [ ] 16 - Guardrails & Safety
- [ ] 17 - Orchestration Frameworks
- [ ] 18 - Production & Deployment
- [ ] 19 - Capstone: Manufacturing Task Descriptions
