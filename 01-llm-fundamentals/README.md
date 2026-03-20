# Module 01: LLM Fundamentals

**Time:** ~60 minutes
**What you'll learn:** How LLMs generate text, what tokens are, what context windows and temperature do, and why any of this matters for your manufacturing project.

**Prerequisites:** Module 00 complete, Ollama running, `qwen3:8b` pulled.

---

## Part 1: Your first conversation with an LLM

Let's not start with theory. Let's start by doing.

Open your terminal and run this:

```bash
ollama run qwen3:8b "What is 2+2?"
```

You just sent a question to a Large Language Model running on your machine, and it answered. No internet, no API key, no subscription.

Now try this one:

```bash
ollama run qwen3:8b "What is the capital of France?"
```

Easy. It knows facts. Now let's try something more interesting:

```bash
ollama run qwen3:8b "Write a one-sentence job task description for a machine operator who inspects weld joints."
```

Look at that -- it generated a professional-sounding task description. But here's the important question: **how did it do that?**

It didn't look up a database of task descriptions. It didn't copy one from the internet. It *generated* that sentence one piece at a time, predicting "what word is most likely to come next?"

That's the core idea behind every LLM. Let's dig into it.

---

## Part 2: How text generation actually works

An LLM does one thing: **given some text, predict the most likely next piece of text.**

That's it. Every impressive thing you've seen an LLM do -- writing code, answering questions, summarizing documents -- is a sophisticated version of "what comes next?"

Let's watch this in action. Run this:

```bash
ollama run qwen3:8b "Complete this sentence with exactly 5 words: The operator must inspect the"
```

Run it again:

```bash
ollama run qwen3:8b "Complete this sentence with exactly 5 words: The operator must inspect the"
```

Did you get the same completion both times? Probably something similar, maybe not identical. We'll explore why in a bit (spoiler: temperature).

Now try giving it a more specific starting point:

```bash
ollama run qwen3:8b "Complete this sentence with exactly 5 words: The welder must verify the"
```

Notice how changing "operator" to "welder" and "inspect" to "verify" shifts what the model predicts next? The model learned patterns from massive amounts of text. It knows that welders typically verify things like "joint integrity" or "weld penetration depth," while generic operators might inspect "finished parts" or "assembly line output."

This is called **autoregressive generation** -- each word depends on all the words before it. The model builds its response one token at a time, and each new token is influenced by everything that came before.

---

## Part 3: Tokens -- what the model actually sees

Here's a key thing: LLMs don't read words. They read **tokens**.

A token is a chunk of text -- sometimes a whole word, sometimes part of a word, sometimes just punctuation. Let's see this for ourselves.

Run this:

```bash
ollama run qwen3:8b "Break the word 'manufacturing' into syllables."
```

The model can do that because it sees "manufacturing" as something like: `["man", "uf", "act", "uring"]` -- roughly 4 tokens. Short common words like "the" or "is" are usually 1 token each.

Why does this matter? Two big reasons:

**1. You pay (or compute) per token, not per word.** Even with local models, more tokens = slower generation.

**2. Context windows are measured in tokens.** We'll get to that in a minute.

Let's build some intuition. Create this small script:

```python
# 01-llm-fundamentals/ex1_token_intuition.py
import ollama

examples = [
    "the",
    "inspect",
    "manufacturing",
    "WPS-201-Rev.C",
    "PPE",
]

for word in examples:
    chars = len(word)
    approx_tokens = max(1, chars // 4)
    print(f"  '{word}' -- {chars} chars, ~{approx_tokens} token(s)")
```

Run it:

```bash
python 01-llm-fundamentals/ex1_token_intuition.py
```

This uses a rough rule of thumb: in English, 1 token is roughly 4 characters. Notice how "the" is about 1 token, but "WPS-201-Rev.C" is about 3-4 tokens. Technical jargon and special formatting eat up more tokens than plain English.

**Try adding your own words to the list.** What about a long specification number from your manufacturing world? A chemical formula? An abbreviation?

---

## Part 4: Temperature -- creativity vs. consistency

This is one of the most practical concepts you'll learn. Temperature controls how "random" the model's word choices are.

Let's see it in action. Run this exact command three times:

```bash
ollama run qwen3:8b "Write one sentence describing a weld inspection task."
```

```bash
ollama run qwen3:8b "Write one sentence describing a weld inspection task."
```

```bash
ollama run qwen3:8b "Write one sentence describing a weld inspection task."
```

You probably got somewhat different answers each time. That's because Ollama uses a default temperature above zero -- the model has some randomness in its choices.

Now let's take control. We can't set temperature from the `ollama run` command line easily, so let's switch to a tiny Python script. This is where things get interesting.

```python
# 01-llm-fundamentals/ex2_temperature.py
import ollama

prompt = "Write one sentence describing a weld inspection task."

print("=== Temperature 0.0 (deterministic) ===")
for i in range(3):
    r = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    print(f"  Run {i+1}: {r['message']['content'].strip()[:100]}")
```

Run it:

```bash
python 01-llm-fundamentals/ex2_temperature.py
```

All three runs should be identical (or nearly identical). At temperature 0.0, the model always picks the single most probable next token. It's completely deterministic.

Now let's see what happens when we crank it up. Create this:

```python
# 01-llm-fundamentals/ex3_temperature_high.py
import ollama

prompt = "Write one sentence describing a weld inspection task."

print("=== Temperature 1.5 (very creative) ===")
for i in range(3):
    r = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 1.5},
    )
    print(f"  Run {i+1}: {r['message']['content'].strip()[:100]}")
```

```bash
python 01-llm-fundamentals/ex3_temperature_high.py
```

See the difference? At 1.5, the outputs are all over the place. Some might even be a little weird or incoherent. The model is taking bigger risks with its word choices.

**Here's the practical takeaway for your manufacturing project:** When you're generating task descriptions that need to be consistent and professional, you want temperature between 0.0 and 0.3. Save higher temperatures for brainstorming or creative tasks.

Let's make this really concrete. Try this one:

```python
# 01-llm-fundamentals/ex4_temperature_comparison.py
import ollama

prompt = "Write one sentence describing a weld inspection task."
temps = [0.0, 0.3, 0.7, 1.0, 1.5]

for temp in temps:
    r = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temp},
    )
    text = r["message"]["content"].strip()
    print(f"  temp={temp}: {text[:90]}")
```

```bash
python 01-llm-fundamentals/ex4_temperature_comparison.py
```

Run it a couple of times. Notice how the low-temperature outputs barely change, while the high-temperature ones shift around every time.

---

## Part 5: Top-p (nucleus sampling)

Temperature isn't the only knob. There's also **top-p**, sometimes called "nucleus sampling."

Here's the idea: instead of adjusting how random the model is across all possible next tokens, top-p says "only consider the most probable tokens that together add up to P% of the probability."

- `top_p=0.9` -- consider the top 90% most likely tokens (wide range of choices)
- `top_p=0.1` -- only consider the very most likely tokens (narrow, focused)

Let's see it:

```python
# 01-llm-fundamentals/ex5_top_p.py
import ollama

prompt = "Write one sentence describing a weld inspection task."

for top_p in [0.1, 0.5, 0.9]:
    r = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"top_p": top_p, "temperature": 0.8},
    )
    text = r["message"]["content"].strip()
    print(f"  top_p={top_p}: {text[:90]}")
```

```bash
python 01-llm-fundamentals/ex5_top_p.py
```

**Rule of thumb:** Adjust temperature OR top-p, not both at the same time. For most work, temperature is the simpler knob to use. Start there.

---

## Part 6: The context window

Every LLM has a limit on how much text it can "see" at once. This is the **context window**, measured in tokens.

Think of it like a desk. You can only spread out so many documents before things start falling off the edge. The context window is the size of that desk.

For `qwen3:8b`, the context window is 128K tokens by default, but in practice, Ollama often defaults to a smaller window for speed. Let's see what this means concretely.

Run this:

```bash
ollama run qwen3:8b "I'm going to tell you a secret code: BLUE-FALCON-42. Remember it. What's 2+2?"
```

It answers the math question. Now ask:

```bash
ollama run qwen3:8b "What was the secret code I told you?"
```

It has no idea. **Why?** Because each `ollama run` command is a completely separate conversation. The model has no memory between calls. Every time you send a request, the model only sees what you include in that specific request.

This is a fundamental concept. Let's see it more clearly:

```python
# 01-llm-fundamentals/ex6_no_memory.py
import ollama

# First call: give it information
r1 = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Remember: the spec number is WPS-201."}],
)
print(f"Response 1: {r1['message']['content'].strip()[:80]}")

# Second call: ask about that information
r2 = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "What spec number did I just mention?"}],
)
print(f"Response 2: {r2['message']['content'].strip()[:80]}")
```

```bash
python 01-llm-fundamentals/ex6_no_memory.py
```

The model doesn't remember. Each API call is a fresh start.

Now let's fix that by putting both messages in the same conversation:

```python
# 01-llm-fundamentals/ex7_conversation.py
import ollama

messages = [
    {"role": "user", "content": "Remember: the spec number is WPS-201."},
    {"role": "assistant", "content": "Got it, the spec number is WPS-201."},
    {"role": "user", "content": "What spec number did I just mention?"},
]

r = ollama.chat(model="qwen3:8b", messages=messages)
print(f"Response: {r['message']['content'].strip()[:80]}")
```

```bash
python 01-llm-fundamentals/ex7_conversation.py
```

Now it remembers, because the earlier messages are included in the context window. The model "remembers" only because we're sending the entire conversation history every time.

**Why this matters for RAG:** When you build a RAG system for manufacturing task descriptions, you won't paste your entire 500-page manual into the prompt. It won't fit (or it'll be slow and expensive). Instead, you'll:
1. Break docs into small chunks
2. Find the relevant chunks for a question
3. Put only those chunks in the prompt

That's what Modules 05-08 are all about.

---

## Part 7: Putting it together

Let's build a slightly bigger script that combines what you've learned. This one asks the model to generate manufacturing task descriptions with different settings so you can see how the parameters interact.

```python
# 01-llm-fundamentals/ex8_combined.py
"""Experiment with LLM parameters for manufacturing task descriptions."""
import ollama


def generate(prompt, temperature=0.0, top_p=1.0):
    """Generate a response with specific settings."""
    r = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "top_p": top_p},
    )
    return r["message"]["content"].strip()


# --- Experiment 1: Consistency for task descriptions ---
print("=== Experiment 1: Consistent task descriptions ===")
print("(temperature=0.0 -- what you'd use in production)\n")

task_prompt = (
    "Write a brief task description for a manufacturing operator "
    "who needs to perform a visual inspection of welded joints "
    "on a steel frame assembly. Include safety requirements."
)

for i in range(3):
    result = generate(task_prompt, temperature=0.0)
    print(f"Run {i+1}:")
    print(f"  {result[:150]}...")
    print()


# --- Experiment 2: Brainstorming mode ---
print("=== Experiment 2: Brainstorming task variations ===")
print("(temperature=0.9 -- for exploring different phrasings)\n")

for i in range(3):
    result = generate(task_prompt, temperature=0.9)
    print(f"Variation {i+1}:")
    print(f"  {result[:150]}...")
    print()


# --- Experiment 3: How context changes output ---
print("=== Experiment 3: Context matters ===\n")

prompts = [
    "Write a task description for inspecting welds.",
    "Write a task description for inspecting welds on a pressure vessel per ASME Section IX.",
    "Write a task description for inspecting welds on a pressure vessel per ASME Section IX. "
    "The operator has Level II VT certification. Include hold points.",
]

for p in prompts:
    result = generate(p, temperature=0.0)
    print(f"Prompt length: {len(p)} chars")
    print(f"  {result[:120]}...")
    print()

print("Notice how more specific context produces more specific output.")
print("This is exactly why RAG works -- better context = better answers.")
```

```bash
python 01-llm-fundamentals/ex8_combined.py
```

Take a few minutes to read through the output. Here's what to pay attention to:

- **Experiment 1:** The three runs at temperature 0.0 should be nearly identical. This is the consistency you want for generating official task descriptions.
- **Experiment 2:** The three runs at temperature 0.9 should be noticeably different. Useful when you want to explore different ways to phrase something.
- **Experiment 3:** More detail in the prompt leads to more specific output. This is the whole premise behind RAG -- by retrieving relevant documents and including them in the prompt, you give the model better context and get better answers.

---

## Try these on your own

Before moving on, try a few experiments. No scripts needed -- just terminal one-liners:

**1. Ask it something it can't know:**

```bash
ollama run qwen3:8b "What is specification WPS-9999-XYZ at Acme Manufacturing?"
```

It will make something up confidently. This is called "hallucination" -- the model generates plausible-sounding text even when it doesn't have real information. This is a huge reason why RAG matters: you ground the model's answers in your actual documents.

**2. Ask it to follow a format:**

```bash
ollama run qwen3:8b "List exactly 3 safety hazards of MIG welding. Use numbered list format."
```

LLMs are generally good at following formatting instructions. You'll use this a lot in Module 04 (Structured Output).

**3. Ask it something ambiguous:**

```bash
ollama run qwen3:8b "Describe the process."
```

Vague input = vague output. Compare that to:

```bash
ollama run qwen3:8b "Describe the process of performing a dye penetrant test on a stainless steel weld joint."
```

Night and day difference. The specificity of your prompt matters enormously.

---

## Takeaways

Here's what you now understand:

1. **LLMs predict the next token** -- every response is built one piece at a time, each piece influenced by everything before it.

2. **Tokens are not words** -- technical jargon, spec numbers, and special formatting use more tokens than plain English. This matters for performance and context window budgets.

3. **Temperature controls consistency** -- 0.0 for deterministic output (task descriptions, documentation), higher values for creative exploration. For manufacturing work, stay in the 0.0 to 0.3 range.

4. **Top-p is another randomness knob** -- use one or the other, not both. Start with temperature.

5. **Context windows are finite** -- the model can only "see" what you put in the current request. No memory between calls. This is exactly why RAG exists: you selectively include relevant information.

6. **More specific prompts produce more specific output** -- this is why retrieving the right documents matters so much in a RAG system.

7. **LLMs hallucinate** -- they will confidently generate plausible-sounding answers even when they don't have real information. Grounding answers in your actual documents (RAG) is how you fix this.

---

## What's next: Module 02

You understand how LLMs work at a fundamental level. Now let's get practical with **running and comparing different local models**. Not all models are created equal -- some are better at following instructions, some at structured output, some at longer text. Module 02 teaches you to pick the right model for the job.

Head over to `02-running-local-llms/` when you're ready.
