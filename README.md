# AI & RAG Learning Journey

This isn't a textbook. It's a hands-on workshop.

You'll start every module by running something small. Then you'll poke at it, change things, see what breaks, and build up from there. By the end of each module, you'll have built something real -- not just read about it.

The goal: build a production-grade AI system that generates manufacturing task descriptions, grounded in your company's actual SOPs and specifications, with evaluation that proves it works and guardrails that catch it when it doesn't.

## Who This Is For

You're a developer or technical professional who needs to confidently build, evaluate, and deploy LLM-powered systems. You need to know not just how to get an LLM to generate content, but how to **prove** it's generating *good* content -- and catch it when it isn't.

## The Core Philosophy

> "If you can't measure it, you can't improve it."

Every module teaches you a capability **and** how to evaluate that capability. Testing isn't a separate phase -- it's woven into everything. By the end, you'll be able to walk into a meeting and say "the AI-generated task descriptions are 23% more consistent than the manually written ones, and here's the evaluation framework that proves it."

## Your Hardware

- **MacBook M4 Pro, 48GB RAM** -- plenty for running local LLMs up to 70B parameters
- **Ollama** as the primary local LLM runtime
- **Recommended models**:
  - **Gemma 3 12B** (`gemma3:12b`) -- Google's Gemma 3 model, course workhorse for modules 00–13
  - **Llama 3.3 70B** (`llama3.3:70b`) -- Meta's latest Llama model, recommended for modules 14+
  - **Llama 4 Scout** -- Meta's newest architecture, good reasoning
  - **Mistral 7B** -- fast and lightweight
- No GPU server needed until fine-tuning (Module 14), and even then we have local options

## How Each Module Works

Every module follows an interactive, workshop-style format:

1. **Start small.** Run a tiny command or a few lines of code. See what happens.
2. **Explain inline.** Right after each step, you'll learn what just happened and why it matters.
3. **Build up gradually.** Each step adds one concept on top of what you just did.
4. **Try, break, fix.** You'll be prompted to change things, test edge cases, and see what breaks.
5. **Bigger exercises at the end.** Once you understand the pieces, you'll combine them into something complete.

Code comes in small pieces (3-10 lines at a time), not giant blocks. You'll see phrases like "run this", "what do you see?", "notice how...", and "what happens if..." throughout.

The tone is conversational. Think of it as pair programming with someone who's done this before.

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
*THE critical skill -- prove your system works and catch regressions*

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
| [19](./19-capstone-manufacturing-tasks/) | **Capstone: Manufacturing Task Descriptions** | Full production system: RAG + evaluation + guardrails + monitoring |

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

- [x] 00 - Environment Setup
- [x] 01 - LLM Fundamentals
- [x] 02 - Running Local LLMs
- [x] 03 - Prompt Engineering
- [x] 04 - Structured Output
- [x] 05 - Embeddings & Vector Stores
- [x] 06 - RAG Fundamentals
- [x] 07 - Advanced RAG
- [x] 08 - Document Processing
- [x] 09 - Evaluation Fundamentals
- [x] 10 - RAG Evaluation with Ragas
- [x] 11 - Testing with DeepEval
- [x] 12 - Observability with Langfuse
- [x] 13 - Evaluation Datasets & Benchmarks
- [x] 14 - Fine-tuning
- [x] 15 - Agents & Tool Use
- [x] 16 - Guardrails & Safety
- [x] 17 - Orchestration Frameworks
- [ ] 18 - Production & Deployment
- [ ] 19 - Capstone: Manufacturing Task Descriptions
