# Module 02: Running Local LLMs

## Goal
Master Ollama as your local LLM runtime. Learn to compare models, understand their tradeoffs, and pick the right model for the right job.

---

## Concepts

### Why Local LLMs?

| Factor | Local (Ollama) | Cloud (OpenAI/Anthropic) |
|--------|---------------|-------------------------|
| **Cost** | Free after hardware | Per-token pricing |
| **Privacy** | Data never leaves your machine | Data sent to provider |
| **Speed** | Depends on hardware (your M4 Pro is great) | Depends on network + server load |
| **Quality** | Good for 7-32B models | Best models available (GPT-4, Claude) |
| **Availability** | Always on, no rate limits | Subject to outages, rate limits |

**For learning:** Local is perfect. No bills, no rate limits, instant feedback.
**For production:** You'll likely use a mix — local for development/testing, cloud for quality-critical tasks.

### The Ollama API

Ollama exposes a REST API (default: `http://localhost:11434`) that's compatible with the OpenAI API format. This means code you write against Ollama can often switch to OpenAI/Anthropic with minimal changes.

### Model Selection for Your M4 Pro (48GB)

| Model | Params | RAM Usage | Speed | Quality | Best For |
|-------|--------|-----------|-------|---------|----------|
| phi3:mini | 3.8B | ~2GB | Very fast | Basic | Quick experiments |
| llama3.1:8b | 8B | ~5GB | Fast | Good | Daily development |
| mistral:7b | 7B | ~4GB | Fast | Good | Structured tasks |
| qwen2.5:14b | 14B | ~9GB | Medium | Great | Quality work |
| qwen2.5:32b | 32B | ~20GB | Slower | Excellent | Best local quality |

**Rule of thumb:** For learning, use 8B models (fast iteration). For your capstone quality work, use 14B-32B.

---

## Exercise 1: Ollama API Exploration

```python
# 02-running-local-llms/ex1_ollama_api.py
"""Learn the Ollama Python API."""

import ollama

# List available models
print("=== Available Models ===")
models = ollama.list()
for model in models.models:
    size_gb = model.size / (1024**3)
    print(f"  {model.model}: {size_gb:.1f}GB")

# Basic chat completion
print("\n=== Basic Chat ===")
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "system",
            "content": "You are a manufacturing process engineer. Be concise and technical.",
        },
        {
            "role": "user",
            "content": "Write a task description for inspecting a CNC-machined part for dimensional accuracy.",
        },
    ],
)
print(response["message"]["content"])

# Streaming — see tokens arrive in real time
print("\n=== Streaming Response ===")
stream = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "user", "content": "List 3 common weld defects in one sentence each."}
    ],
    stream=True,
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()

# Raw generation (no chat format) — useful for understanding the difference
print("\n=== Raw Generate vs Chat ===")
raw = ollama.generate(
    model="llama3.1:8b",
    prompt="Complete this manufacturing task description: 'The operator shall'",
)
print(f"Raw completion: {raw['response'][:200]}")
```

---

## Exercise 2: Model Comparison Tool

```python
# 02-running-local-llms/ex2_model_compare.py
"""Compare how different models handle the same prompt."""

import ollama
import time

PROMPT = """Write a professional task description for an assembly line operator
who needs to install a circuit board into a housing unit. Include safety
requirements and quality checks. Keep it under 100 words."""

# Test with available models — adjust this list based on what you've pulled
models_to_test = ["llama3.1:8b", "phi3:mini"]

results = {}

for model_name in models_to_test:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    try:
        start = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT}],
            options={"temperature": 0.1},
        )
        elapsed = time.time() - start

        content = response["message"]["content"]
        word_count = len(content.split())

        results[model_name] = {
            "content": content,
            "time": elapsed,
            "words": word_count,
        }

        print(f"Time: {elapsed:.1f}s")
        print(f"Words: {word_count}")
        print(f"Response:\n{content}")

    except Exception as e:
        print(f"Error: {e}")
        print(f"(Pull this model first: ollama pull {model_name})")

# Summary comparison
print(f"\n{'='*60}")
print("COMPARISON SUMMARY")
print(f"{'='*60}")
for model_name, data in results.items():
    print(f"{model_name:20s} | {data['time']:5.1f}s | {data['words']:3d} words")

print("\n💡 Notice: faster isn't always better. Look at QUALITY, not just speed.")
print("   Does it follow the prompt? Is it professional? Under 100 words?")
print("   This manual evaluation is exactly what we'll automate in Modules 09-13.")
```

---

## Exercise 3: The OpenAI-Compatible API

```python
# 02-running-local-llms/ex3_openai_compat.py
"""Use Ollama through the OpenAI-compatible API.
   This matters because it means your code can easily swap between local and cloud."""

from openai import OpenAI

# Point the OpenAI client at your local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a key, but the client requires one
)

# Same OpenAI API format, running locally
response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[
        {
            "role": "system",
            "content": "You are a technical writer for manufacturing documentation.",
        },
        {
            "role": "user",
            "content": "Write a safety precaution for operating a hydraulic press.",
        },
    ],
    temperature=0.2,
)

print("=== Response via OpenAI-compatible API ===")
print(response.choices[0].message.content)

print("\n=== Why This Matters ===")
print("Your code works with Ollama locally, but by changing just the")
print("base_url and api_key, you can switch to OpenAI, Azure, or any")
print("OpenAI-compatible provider. Write once, deploy anywhere.")

# To switch to real OpenAI (when you need it):
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# response = client.chat.completions.create(model="gpt-4o", ...)
```

---

## Exercise 4: Understanding Model Parameters

```python
# 02-running-local-llms/ex4_parameters.py
"""Explore model parameters and their effects."""

import ollama

PROMPT = "Describe the process of TIG welding aluminum in exactly 2 sentences."

# Parameter experiments
experiments = {
    "default": {},
    "low_temp": {"temperature": 0.0},
    "high_temp": {"temperature": 1.5},
    "short_output": {"temperature": 0.0, "num_predict": 50},  # max tokens
    "long_context": {"temperature": 0.0, "num_ctx": 4096},    # context window size
    "repetition_penalty": {"temperature": 0.3, "repeat_penalty": 1.5},
}

for name, options in experiments.items():
    print(f"\n--- {name} (options: {options}) ---")
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": PROMPT}],
        options=options,
    )
    print(response["message"]["content"][:200])

print("\n=== Key Parameters Reference ===")
print("temperature   : 0.0-2.0  — randomness of output")
print("num_predict   : int      — max tokens to generate (like max_tokens)")
print("num_ctx       : int      — context window size")
print("top_p         : 0.0-1.0  — nucleus sampling threshold")
print("repeat_penalty: float    — penalize repeated phrases (useful for task descriptions)")
print("stop          : [str]    — stop generating at these strings")
```

---

## Takeaways

1. **Ollama is your local LLM powerhouse** — free, private, always available
2. **Model size is a tradeoff** — bigger = better quality but slower; 8B is great for development
3. **The OpenAI-compatible API** means your code is portable between local and cloud
4. **Parameters matter** — temperature and repeat_penalty are especially important for consistent manufacturing documentation
5. **You need to manually evaluate quality now** — "this looks good" isn't good enough for production (we fix this in Phase 3)

## Setting the Stage for Module 03

You can run models and generate text. But garbage in = garbage out. **Prompt engineering** is the art of getting LLMs to do exactly what you want. Module 03 teaches you systematic techniques — not tricks — for writing prompts that produce consistent, professional-quality output. This is where the manufacturing task description work really begins.
