# Module 02: Running Local LLMs

## Goal
Get your hands on Ollama -- your local LLM runtime. By the end of this module, you'll be chatting with models from the command line, calling them from Python, swapping between models, and tuning parameters to control output. All running on your machine, no cloud bills.

---

## Part 1: Your First Conversation with a Local Model

Let's start in the terminal. No Python yet -- just you and the model.

### Step 1: See what you have

Open your terminal and run:

```bash
ollama list
```

What do you see? You should get a table showing models you've already pulled. Look for `qwen3:8b` -- that's your workhorse for this journey.

If it's there, great. If not:

```bash
ollama pull qwen3:8b
```

This downloads about 5GB. On your M4 Pro with 48GB RAM, this model fits easily -- it uses about 5GB when running.

### Step 2: Talk to it

```bash
ollama run qwen3:8b
```

You're now in interactive chat mode. You've got a blinking cursor. Let's try something from your world:

```
What are the 3 most common weld defects in structural steel fabrication?
```

Look at the response. Notice how it gives you a pretty reasonable answer? That's a 32-billion-parameter model running *entirely on your laptop*. No internet needed, no API key, no bill at the end of the month.

### Step 3: Try something harder

Still in the same chat session, type:

```
Write a task description for a manufacturing operator who needs to inspect incoming steel plates for surface defects.
```

Read the output carefully. Ask yourself:
- Is it professional enough for your facility?
- Does it include the right kind of detail?
- Would you hand this to an operator as-is?

Probably not perfect -- and that's OK. We'll fix that in Module 03.

### Step 4: Test its limits

Try this:

```
What is specification AWS D1.1 Section 6.10.2 about?
```

It might give you something that *sounds* right but is partially or completely made up. This is called **hallucination** -- the model generates plausible-sounding text that isn't grounded in fact. Remember this. It matters a LOT for manufacturing documentation, and it's exactly why we'll build RAG (retrieval-augmented generation) later.

Type `/bye` to exit the chat.

### Step 5: Try a smaller model

Let's pull a much smaller model to compare:

```bash
ollama pull phi3:mini
```

Once it's done:

```bash
ollama run phi3:mini
```

Ask it the exact same question:

```
Write a task description for a manufacturing operator who needs to inspect incoming steel plates for surface defects.
```

Compare this output to what qwen3:8b gave you. Notice the difference? The smaller model (3.8B parameters) is faster but the output is usually less detailed, less professional, and more likely to miss important things like safety requirements or documentation steps.

Type `/bye` to exit.

That tradeoff -- speed vs. quality -- is something you'll navigate throughout this journey.

---

## Part 2: Calling Ollama from Python

The command line is great for exploring. But for real work, you need Python. Let's switch.

### Step 6: Your first Python call

Create a file called `try_chat.py` in this module's directory:

```python
# 02-running-local-llms/try_chat.py
import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {"role": "user", "content": "Name 3 types of non-destructive testing used in manufacturing."}
    ],
)

print(response["message"]["content"])
```

Run it:

```bash
python 02-running-local-llms/try_chat.py
```

What do you see? You should get a response about NDT methods -- maybe ultrasonic, radiographic, and magnetic particle testing.

Notice the structure: you send a list of `messages`, each with a `role` and `content`. The model sends back a response in the same format. That's the **chat completion** pattern, and it's the standard across almost every LLM API.

### Step 7: Add a system prompt

Now let's tell the model *who it is*. Edit your file:

```python
# 02-running-local-llms/try_chat.py
import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {"role": "system", "content": "You are a manufacturing process engineer. Be concise and technical."},
        {"role": "user", "content": "Name 3 types of non-destructive testing used in manufacturing."},
    ],
)

print(response["message"]["content"])
```

Run it again. See the difference? The system prompt steers the model's personality and style. We'll go deep on this in Module 03, but notice it already -- same question, different framing, different output.

### Step 8: Try generate() instead of chat()

There's another way to call Ollama -- raw text completion with no chat structure:

```python
# 02-running-local-llms/try_generate.py
import ollama

result = ollama.generate(
    model="qwen3:8b",
    prompt="Complete this sentence: The operator shall inspect each weld joint by",
)

print(result["response"])
```

Run it. See how it just *continues* the text? No system prompt, no message history -- just raw completion. This is closer to how LLMs actually work under the hood. The chat format (`ollama.chat()`) is a convenience wrapper that most people use.

For our manufacturing work, `ollama.chat()` is almost always what you want. But it's good to know `generate()` exists.

### Step 9: Watch it think -- streaming

When a model generates a long response, you don't have to wait for the whole thing. You can watch tokens arrive in real time:

```python
# 02-running-local-llms/try_streaming.py
import ollama

stream = ollama.chat(
    model="qwen3:8b",
    messages=[
        {"role": "user", "content": "List 5 common safety hazards in a machine shop. One sentence each."}
    ],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()
```

Run it. See the words appearing one by one? That's streaming. It's the same experience as ChatGPT's typing effect. For your manufacturing tool, streaming will make the UI feel responsive even when generating longer task descriptions.

---

## Part 3: The OpenAI-Compatible API

This is one of Ollama's best features, and it matters more than you might think.

### Step 10: Same code, different client

Ollama speaks the same API language as OpenAI. That means you can use the official `openai` Python library to talk to your local model:

```python
# 02-running-local-llms/try_openai_compat.py
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the library requires something
)

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[
        {"role": "system", "content": "You are a technical writer for manufacturing documentation."},
        {"role": "user", "content": "Write a safety precaution for operating a hydraulic press."},
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)
```

Run it. The response comes from your local `qwen3:8b`, but the code looks *exactly* like it would for calling OpenAI's GPT-4.

### Why does this matter?

Think about it this way: you build your manufacturing tool using this `openai` client format. Later, if you want to switch to GPT-4 for higher quality, you change **two lines**:

```python
# Local (Ollama):
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(model="qwen3:8b", ...)

# Cloud (OpenAI):
client = OpenAI(api_key="sk-your-real-key-here")
response = client.chat.completions.create(model="gpt-4o", ...)
```

Same code structure. Same message format. You develop locally for free, deploy to cloud when you need top quality. This is called **code portability**, and it saves you from rewriting everything when requirements change.

---

## Part 4: Model Parameters -- Controlling the Output

The model doesn't just generate text randomly. You can tune *how* it generates text. Let's experiment.

### Step 11: Temperature -- the creativity knob

Temperature controls randomness. Low = predictable. High = creative (and sometimes weird).

```python
# 02-running-local-llms/try_temperature.py
import ollama

prompt = "Describe the process of TIG welding aluminum in exactly 2 sentences."

for temp in [0.0, 0.5, 1.0, 1.5]:
    response = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temp},
    )
    print(f"\n--- temperature={temp} ---")
    print(response["message"]["content"])
```

Run it. Compare the outputs at each temperature.

Notice:
- At `0.0`, the output is deterministic -- run it twice and you'll get nearly the same thing
- At `0.5`, there's slight variation but it stays on topic
- At `1.0`, it gets more creative (the default)
- At `1.5`, it might start rambling or getting weird

**For manufacturing documentation, you almost always want temperature between 0.0 and 0.3.** You want consistency and accuracy, not creativity.

### Step 12: num_predict -- limiting output length

What if you want to cap how much the model writes?

```python
# 02-running-local-llms/try_num_predict.py
import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Explain lockout/tagout procedures."}],
    options={"num_predict": 50},
)

print(response["message"]["content"])
print(f"\n(Output was limited to ~50 tokens)")
```

Run it. The response gets cut off around 50 tokens. This is useful when you need short, punchy outputs -- like a one-line task summary. `num_predict` is Ollama's equivalent of OpenAI's `max_tokens`.

### Step 13: repeat_penalty -- stopping the broken record

LLMs sometimes get stuck in loops, repeating phrases. This is especially annoying in structured documentation. Try this:

```python
# 02-running-local-llms/try_repeat_penalty.py
import ollama

prompt = "List 10 steps for a machine startup checklist."

for penalty in [1.0, 1.2, 1.5]:
    response = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "repeat_penalty": penalty},
    )
    print(f"\n--- repeat_penalty={penalty} ---")
    print(response["message"]["content"][:400])
```

Run it. At `1.0` (no penalty), you might see repetitive language across steps. At `1.2`, the model tries harder to vary its phrasing. At `1.5`, it actively avoids repetition -- sometimes too aggressively.

**For task descriptions, `1.1` to `1.3` is the sweet spot.** Enough to prevent "Ensure safety. Ensure quality. Ensure compliance." loops, but not so much that the output gets strange.

### Quick Reference: Key Parameters

| Parameter | Range | What It Does | Good Value for Mfg Docs |
|-----------|-------|-------------|------------------------|
| `temperature` | 0.0 - 2.0 | Controls randomness | 0.1 - 0.3 |
| `num_predict` | integer | Max tokens to generate | 200 - 500 |
| `num_ctx` | integer | Context window size | 4096 (default is fine) |
| `top_p` | 0.0 - 1.0 | Nucleus sampling threshold | 0.9 |
| `repeat_penalty` | float | Penalizes repeated phrases | 1.1 - 1.3 |
| `stop` | list of strings | Stop generating at these | Depends on format |

---

## Part 5: Which Model Should You Use?

Your M4 Pro with 48GB RAM can run some seriously capable models. Here's your cheat sheet:

| Model | Params | RAM Used | Speed on M4 Pro | Quality | When to Use |
|-------|--------|----------|-----------------|---------|-------------|
| `phi3:mini` | 3.8B | ~2GB | Very fast | Basic | Quick experiments, testing code logic |
| `qwen3:8b` | 8B | ~5GB | Fast | Good | **Daily learning and development (modules 00–13)** |
| `qwen3:32b` | 32B | ~20GB | Moderate | Excellent | Higher quality for advanced modules (14+) |
| `llama4:scout` | 109B (MoE) | ~30GB | Slow | Excellent | Best local quality (mixture of experts) |

**Recommendation:** Use `qwen3:8b` for all exercises in modules 00–13. It's fast, has no extended thinking overhead, and is more than sufficient for learning exercises. When you reach modules 14+, switch to `qwen3:32b` for the extra quality those advanced topics benefit from.

Want to try the larger model for comparison?

```bash
ollama pull qwen3:32b
```

The larger model produces higher quality output but is noticeably slower. Try it on the same manufacturing prompt later and compare.

---

## Part 6: Putting It Together -- Model Comparison Exercise

Now let's do something more substantial. You're going to compare two models on the same manufacturing prompt and measure the results.

```python
# 02-running-local-llms/ex_model_compare.py
"""Compare two models on a manufacturing task description prompt."""

import ollama
import time

PROMPT = """Write a professional task description for an assembly line operator
who needs to install a circuit board into a housing unit. Include safety
requirements and quality checks. Keep it under 100 words."""

models_to_test = ["qwen3:8b", "phi3:mini"]

results = {}

for model_name in models_to_test:
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

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

        print(f"Time: {elapsed:.1f}s | Words: {word_count}")
        print(f"\n{content}")

    except Exception as e:
        print(f"Error: {e}")
        print(f"(You may need to pull this model: ollama pull {model_name})")

# Summary
print(f"\n{'=' * 60}")
print("COMPARISON SUMMARY")
print(f"{'=' * 60}")
print(f"{'Model':<20} {'Time':>6} {'Words':>6}")
print(f"{'-'*20} {'-'*6} {'-'*6}")
for model_name, data in results.items():
    print(f"{model_name:<20} {data['time']:>5.1f}s {data['words']:>5d}")

print("\nLook at both outputs and ask yourself:")
print("  - Which one followed the 'under 100 words' instruction?")
print("  - Which one included safety requirements?")
print("  - Which one sounds like it belongs in a real work instruction?")
print("  - Is the faster model 'good enough' for your use case?")
print("\nThere's no single right answer. It depends on what you need.")
print("For learning and iteration: fast is better (qwen3:8b or phi3:mini).")
print("For quality output: bigger is usually better (qwen3:32b for modules 14+).")
```

Run it:

```bash
python 02-running-local-llms/ex_model_compare.py
```

Read both outputs carefully. This is the kind of evaluation thinking you'll formalize in Modules 09-13, but for now, trust your gut: which output would you actually hand to a supervisor?

---

## Takeaways

1. **Ollama is your local LLM runtime** -- free, private, always available, no API keys needed
2. **`ollama.chat()` is the main Python API** -- send messages, get responses, that's the pattern
3. **The OpenAI-compatible API** means your code works locally now and can switch to cloud later with a two-line change
4. **Temperature and repeat_penalty** are the two parameters that matter most for consistent manufacturing documentation
5. **Model size is a tradeoff** -- 32B for quality during development, smaller models for faster iteration when needed
6. **Models hallucinate** -- they'll make up spec numbers and procedures that sound right but aren't. RAG (Module 06) is how we fix this.

## What's Next

You can run models and get text back. But did you notice the output quality was... OK? Not great? The model doesn't know your facility's style, your format requirements, or your standards. **Prompt engineering** (Module 03) is where you learn to tell the model *exactly* what you want -- and get it consistently.
