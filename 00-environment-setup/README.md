# Module 00: Environment Setup

## Goal
Get your entire development environment ready so every future module "just works." By the end of this module, you'll have Python, Ollama, and all dependencies installed and verified.

---

## Step 1: Python Environment

We use Python 3.11+ with a virtual environment to keep things clean.

```bash
# From the repo root
python3 -m venv .venv
source .venv/bin/activate

# Verify
python --version  # Should be 3.11+
```

**Why a virtual environment?** It isolates this project's dependencies from your system Python. When you're done for the day, just `deactivate`. When you come back, `source .venv/bin/activate`.

---

## Step 2: Install Ollama

Ollama is your local LLM runtime. It runs models directly on your M4 Pro's GPU (via Metal).

```bash
# Install Ollama (if not already installed)
brew install ollama

# Start the Ollama server (runs in background)
ollama serve &

# Pull your first model — Llama 3.1 8B is a great starting point
ollama pull llama3.1:8b

# Also grab a small model for quick experiments
ollama pull phi3:mini

# Verify it works
ollama run llama3.1:8b "Say hello in one sentence"
```

**What just happened?** Ollama downloaded a quantized (compressed) version of Meta's Llama 3.1 model. The 8B parameter version uses ~5GB of RAM. Your 48GB machine can comfortably run models up to ~32B parameters.

### Recommended Models to Pull (you can do this now or as needed)

```bash
# For learning (fast, small)
ollama pull phi3:mini          # 3.8B params, ~2GB RAM
ollama pull llama3.1:8b        # 8B params, ~5GB RAM

# For quality work (your daily driver)
ollama pull llama3.1:latest    # 8B params, good balance
ollama pull mistral:latest     # 7B params, great for structured tasks

# For when you need the best local quality
ollama pull qwen2.5:32b        # 32B params, ~20GB RAM, excellent quality
```

---

## Step 3: Install Python Dependencies

```bash
# From the repo root, with venv activated
pip install -r requirements.txt
```

This installs everything for all modules. Some packages are large (especially `transformers` and `sentence-transformers`), so this may take a few minutes.

---

## Step 4: Environment Variables

```bash
# Copy the example env file
cp .env.example .env
```

**For now, you don't need any API keys.** Modules 00-13 work entirely with local Ollama models. If you later want to compare against cloud models (OpenAI, Anthropic), you can add those keys then.

---

## Step 5: Verify Everything Works

Create and run this verification script:

```python
# 00-environment-setup/verify_setup.py
"""Verify that the learning journey environment is ready."""

import sys
print(f"✓ Python {sys.version}")

# Check key packages
try:
    import ollama
    print("✓ ollama package installed")
except ImportError:
    print("✗ ollama package missing — run: pip install ollama")

try:
    import chromadb
    print("✓ chromadb installed")
except ImportError:
    print("✗ chromadb missing")

try:
    import langchain
    print("✓ langchain installed")
except ImportError:
    print("✗ langchain missing")

try:
    from ragas import evaluate
    print("✓ ragas installed")
except ImportError:
    print("✗ ragas missing")

try:
    from deepeval import evaluate as deep_evaluate
    print("✓ deepeval installed")
except ImportError:
    print("✗ deepeval missing")

try:
    import langfuse
    print("✓ langfuse installed")
except ImportError:
    print("✗ langfuse missing")

# Check Ollama connectivity
try:
    client = ollama.Client()
    models = client.list()
    model_names = [m.model for m in models.models]
    print(f"✓ Ollama running with models: {model_names}")
except Exception as e:
    print(f"✗ Ollama not reachable: {e}")
    print("  Run: ollama serve")

print("\n🎯 If all checks pass, you're ready for Module 01!")
```

```bash
python 00-environment-setup/verify_setup.py
```

---

## Step 6: Understand the Project Structure

```
rag-ai-learning-journey/
├── .venv/                    # Your Python virtual environment
├── .env                      # Your API keys (never committed)
├── requirements.txt          # All Python dependencies
├── CLAUDE.md                 # Project context for AI assistants
├── README.md                 # The master learning path
│
├── 00-environment-setup/     # ← You are here
├── 01-llm-fundamentals/      # How LLMs actually work
├── 02-running-local-llms/    # Ollama deep dive
│   ...
├── 19-capstone-.../          # Final project
```

---

## Takeaways

- You have a **clean Python environment** isolated from your system
- **Ollama** is running locally and can serve LLM models via API
- All **Python packages** are installed for every module
- You don't need cloud API keys — everything works locally on your M4 Pro

## Setting the Stage for Module 01

You've got the tools. Now you need to understand **what an LLM actually is** — not the hype, but the mechanics. How does it generate text? What are tokens? Why does "temperature" matter? Module 01 gives you the mental model that makes everything else click.
