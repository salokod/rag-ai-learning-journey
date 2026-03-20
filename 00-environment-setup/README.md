# Module 00: Environment Setup

**Time:** ~30 minutes
**What you'll have when done:** Python, Ollama, and all dependencies installed and verified. Ready to talk to an LLM running on your own machine.

---

## First things first: open your terminal

Everything in this course happens from the command line. Open Terminal (or iTerm, or whatever you prefer) and navigate to this repo:

```bash
cd ~/repos/rag-ai-learning-journey
```

Let's make sure you're in the right place. Run this:

```bash
ls
```

You should see folders like `00-environment-setup/`, `01-llm-fundamentals/`, `requirements.txt`, etc. If you do, great -- you're in the right spot.

---

## Step 1: Create a Python virtual environment

A virtual environment is like a clean workbench. It keeps this project's tools separate from everything else on your Mac. Think of it like having a dedicated toolbox for one job instead of dumping everything into one drawer.

Run this:

```bash
python3 -m venv .venv
```

Nothing happened? Good. That means it worked. Now activate it:

```bash
source .venv/bin/activate
```

Notice how your terminal prompt changed? You should see `(.venv)` at the beginning of the line. That tells you the virtual environment is active.

Now let's check your Python version:

```bash
python --version
```

You need 3.11 or higher. If you see something like `Python 3.12.x` or `Python 3.13.x`, you're good.

**Quick reference for later:**
- Leaving for the day? Type `deactivate` to exit the venv.
- Coming back? Run `source .venv/bin/activate` again from the repo root.
- Forgot whether it's active? Look for `(.venv)` in your prompt.

---

## Step 2: Install Ollama

Ollama is the engine that runs LLMs locally on your Mac. Your M4 Pro has a powerful GPU built in, and Ollama knows how to use it. No cloud, no API keys, no per-token charges.

Run this:

```bash
brew install ollama
```

If you see "already installed," that's fine -- move on.

Now start the Ollama server:

```bash
ollama serve &
```

That `&` at the end runs it in the background so you can keep using your terminal. You might see some log output -- that's normal.

Let's verify it's running:

```bash
ollama list
```

If this is your first time, the list will be empty. That's expected -- we haven't downloaded any models yet.

If you get "Error: could not connect to ollama," wait a few seconds and try again. The server sometimes needs a moment to start up.

---

## Step 3: Pull your first model

This is the fun part. You're about to download an actual Large Language Model to your machine.

Run this:

```bash
ollama pull qwen3:8b
```

This downloads Alibaba's Qwen 3 model with 8 billion parameters. It's about 5GB, so it will take a minute or two depending on your connection.

While it downloads, here's what's happening: Ollama is pulling a "quantized" (compressed) version of the model. The full model would be much larger, but quantization shrinks it down to fit comfortably in memory while keeping most of the quality. Your 48GB machine handles this easily.

When it finishes, let's make sure it worked:

```bash
ollama list
```

You should see `qwen3:8b` in the list now. Notice the size column -- around 5GB.

Let's give it a spin. Run this:

```bash
ollama run qwen3:8b "Say hello in one sentence."
```

You just ran an LLM on your own hardware. No internet needed, no API key, no usage fees. The model processed your request entirely on your M4 Pro's GPU.

**Try another one:**

```bash
ollama run qwen3:8b "What are the three main types of welding?"
```

See how fast that was? That's the M4 Pro's Metal GPU at work.

---

## Step 4: A note on other models

For modules 00–13 (fundamentals through evaluation), `qwen3:8b` is your workhorse. It's fast, has no extended thinking overhead, and is more than sufficient for the exercises in these modules.

As you get more comfortable, you might want to try other models. Here are a couple worth knowing about:

```bash
ollama pull qwen3:32b
```

Qwen 3 32B is the larger sibling -- higher quality output, but slower and uses more memory. Since you downloaded the course earlier, `qwen3:32b` is likely already on your machine. **It's recommended for modules 14 and later** (fine-tuning, agents, orchestration, production, and the capstone), where the extra quality is worth the wait.

```bash
ollama pull llama4:scout
```

Llama 4 Scout is Meta's newest model architecture. It uses a "mixture of experts" design -- think of it like having several specialists who each handle different kinds of questions, rather than one generalist.

You do NOT need to pull `llama4:scout` right now. But it's good to know your options.

**How to think about model sizes on your machine:**
- 8B parameter models (~5GB) -- fast, no thinking overhead, great for learning exercises
- 32B parameter models (~20GB) -- excellent quality, recommended for modules 14+
- 70B+ models -- won't fit well, skip these for now

---

## Step 5: Install Python dependencies

Make sure your virtual environment is active (you should see `(.venv)` in your prompt), then run:

```bash
pip install -r requirements.txt
```

This installs everything you'll need across all 20 modules. It includes packages for:
- Talking to Ollama from Python (`ollama`)
- Vector databases (`chromadb`)
- RAG frameworks (`langchain`)
- Evaluation tools (`ragas`, `deepeval`)
- And more

This will take a few minutes. Some packages are large (especially `transformers` and `sentence-transformers`).

When it finishes, let's do a quick sanity check:

```bash
python -c "import ollama; print('ollama package works')"
```

You should see `ollama package works`. If you see an error, make sure your venv is active and try `pip install ollama` directly.

---

## Step 6: Set up your environment file

```bash
cp .env.example .env
```

This creates a local `.env` file for API keys. Here's the thing though -- you don't need any API keys right now. Modules 00 through 13 work entirely with local Ollama models. If you later want to compare against cloud models like OpenAI or Anthropic, you can add keys then.

---

## Step 7: Verify everything works

Now let's run the verification script that checks all the pieces at once:

```bash
python 00-environment-setup/verify_setup.py
```

Go through the output line by line. Here's what to look for:

- Lines starting with a checkmark mean that component is ready.
- Lines starting with an X mean something is missing.

**If you see "Ollama not reachable":**

That means the Ollama server isn't running. Open a new terminal tab and run:

```bash
ollama serve
```

Then come back and run the verify script again.

**If you see a missing package:**

Run the `pip install` command shown in the error message. For example:

```bash
pip install chromadb
```

Then run the verify script again.

**If you see "No Qwen model found":**

```bash
ollama pull qwen3:8b
```

**If everything passes:** you're done with setup. Every tool you need for the entire learning journey is installed and working.

---

## Quick tour of the project structure

Run this to see the layout:

```bash
ls -d */
```

You'll see 20 module folders numbered `00-` through `19-`. Each one builds on the previous, so go through them in order.

The key files at the root:
- `requirements.txt` -- all Python dependencies (you already installed these)
- `.env` -- your local API keys (not committed to git)
- `CLAUDE.md` -- project context
- `README.md` -- the master learning path

---

## Takeaways

Here's what you just accomplished:

- **Python virtual environment** -- your isolated workspace for this project
- **Ollama running locally** -- an LLM engine using your M4 Pro's GPU, no cloud needed
- **qwen3:8b downloaded** -- a real 8-billion-parameter model on your machine
- **All Python packages installed** -- ready for every module in the journey
- **Verification script passing** -- everything confirmed working

You ran an LLM on your own hardware. That's not a small thing. Most people only interact with LLMs through a chat window on a website. You now have direct access to the engine itself.

---

## What's next: Module 01

You've got the tools installed. Now you need to understand what an LLM actually *is*. Not the hype, not the marketing -- the actual mechanics. How does it generate text? What are tokens? Why does "temperature" matter?

Module 01 gives you the mental model that makes everything else in this journey click. You'll start by running commands directly in the terminal, then gradually move into Python.

Head over to `01-llm-fundamentals/` when you're ready.
