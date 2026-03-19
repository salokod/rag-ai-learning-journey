# CLAUDE.md

## Project Overview
This is a hands-on AI/RAG learning journey repository. It teaches LLM fundamentals through production-grade evaluation and deployment, building toward a capstone project: an AI-powered manufacturing task description system.

## Structure
- 20 modules (00-19), each in its own directory
- Each module has a README.md with teaching content, concepts, and exercises
- Code examples are in Python, designed to run on macOS with Apple Silicon (M4 Pro, 48GB RAM)
- Primary tools: Ollama (local LLMs), ChromaDB (vectors), Ragas/DeepEval (evaluation), Langfuse (observability)

## Key Conventions
- Python 3.11+ with virtual environment in `.venv`
- All exercises are self-contained within their module directory
- Environment variables go in `.env` (never committed)
- Each module builds on the previous — they should be completed in order

## Running Exercises
```bash
source .venv/bin/activate
cd <module-directory>
python <exercise-file>.py
```

## The Evaluation Thread
Testing and evaluation is the core theme. Every module from 06 onward includes evaluation components, not just the dedicated evaluation modules (09-13).
