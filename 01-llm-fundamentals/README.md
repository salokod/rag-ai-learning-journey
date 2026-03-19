# Module 01: LLM Fundamentals

## Goal
Understand how Large Language Models actually work — not at PhD level, but enough to make smart decisions about prompts, model selection, and debugging when things go wrong.

---

## Concepts

### What Is an LLM?

An LLM (Large Language Model) is a neural network trained on massive amounts of text. Its core ability is deceptively simple: **given some text, predict the most likely next token.**

That's it. Every impressive thing an LLM does — writing code, answering questions, summarizing documents — is a sophisticated version of "what word comes next?"

### Tokens: The Atoms of LLMs

LLMs don't see words. They see **tokens** — chunks of text that might be a word, part of a word, or punctuation.

```
"Manufacturing" → ["Man", "uf", "act", "uring"]     (4 tokens)
"the"           → ["the"]                             (1 token)
"task"          → ["task"]                             (1 token)
```

**Why this matters to you:**
- You pay (or compute) per token, not per word
- Context windows are measured in tokens (e.g., 8K, 32K, 128K tokens)
- Some words use more tokens than others — technical jargon often tokenizes poorly

### Context Window

The context window is **how much text the LLM can "see" at once** — both your input AND its output combined.

```
┌─────────────────────────────────────────┐
│           Context Window (8K tokens)     │
│                                         │
│  [System Prompt]  ~200 tokens           │
│  [Your Input]     ~1000 tokens          │
│  [LLM's Output]   ~800 tokens          │
│  [Remaining]       ~6000 tokens free    │
│                                         │
└─────────────────────────────────────────┘
```

For your manufacturing task descriptions, a small context window (8K) is probably fine. For RAG with many retrieved documents, you'll want larger windows (32K+).

### Temperature: Creativity vs. Consistency

Temperature controls **how random** the model's token selection is.

| Temperature | Behavior | Use Case |
|------------|----------|----------|
| 0.0 | Always picks the most probable token | Factual tasks, structured output |
| 0.3 | Slightly varied, mostly predictable | Task descriptions, documentation |
| 0.7 | Balanced creativity and coherence | General writing |
| 1.0+ | Wild, creative, may be incoherent | Brainstorming, creative writing |

**For your work:** When generating consistent task descriptions, you'll likely want temperature 0.0-0.3.

### Top-p (Nucleus Sampling)

Top-p is another way to control randomness. Instead of adjusting the "temperature" of the distribution, it says "only consider tokens that make up the top P% of probability."

- `top_p=0.9` → Consider the top 90% of probable tokens
- `top_p=0.1` → Only consider the most probable tokens

**Rule of thumb:** Adjust temperature OR top-p, not both. Start with temperature.

### How Generation Actually Works

```
Input: "The task requires the operator to"

Step 1: Model sees input, generates probability distribution:
   "inspect"  → 0.23
   "verify"   → 0.18
   "ensure"   → 0.15
   "check"    → 0.12
   ...

Step 2: Based on temperature, select a token → "inspect"

Step 3: New input becomes: "The task requires the operator to inspect"

Step 4: Generate next token probability distribution:
   "the"      → 0.31
   "all"      → 0.14
   "each"     → 0.12
   ...

Step 5: Repeat until stop condition (max tokens, stop token, etc.)
```

This is called **autoregressive generation** — each token depends on all previous tokens.

---

## Environment

No new setup needed — you already have Ollama and the Python packages from Module 00.

---

## Exercise 1: Explore Tokenization

```python
# 01-llm-fundamentals/ex1_tokenization.py
"""Explore how text gets broken into tokens."""

import ollama


def count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 characters per token for English."""
    return len(text) // 4


# Let's see how different text tokenizes
examples = [
    "The operator must inspect the weld joint.",
    "Manufacturing process control specification #MP-2847-Rev.C",
    "Ensure proper PPE is worn at all times during the grinding operation.",
    "機械加工",  # Japanese: "machining" — non-English uses more tokens
]

print("=== Tokenization Explorer ===\n")
for text in examples:
    approx = count_tokens_approx(text)
    print(f"Text: {text}")
    print(f"  Characters: {len(text)}")
    print(f"  Approx tokens: {approx}")
    print(f"  Ratio: ~{len(text)/max(approx,1):.1f} chars/token")
    print()

# Now let's see it in action — ask the model about its own tokens
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "user",
            "content": "Break this sentence into individual words and count them: "
            "'The operator must verify the torque specification before assembly.'",
        }
    ],
)
print("=== Model's Word Analysis ===")
print(response["message"]["content"])

# Key insight: the model's token count of your input affects
# how much context window is left for retrieval docs (in RAG)
print("\n=== Practical Implication ===")
task_desc = """Inspect all weld joints on assembly #4872 per specification WPS-201.
Verify penetration depth meets minimum 3.2mm requirement.
Document any defects using form QC-107 and photograph with scale reference.
If defects exceed acceptance criteria per AWS D1.1, tag part for rework."""

approx_tokens = count_tokens_approx(task_desc)
print(f"Sample task description: {approx_tokens} tokens")
print(f"In an 8K context window, you could fit ~{8000 // approx_tokens} similar descriptions")
print(f"In a 32K window: ~{32000 // approx_tokens} descriptions")
```

---

## Exercise 2: Temperature Experiments

```python
# 01-llm-fundamentals/ex2_temperature.py
"""See how temperature affects output consistency."""

import ollama

PROMPT = """Write a one-sentence task description for an operator who needs to
inspect a welded joint for defects. Be specific and professional."""

temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]

for temp in temperatures:
    print(f"\n{'='*60}")
    print(f"Temperature: {temp}")
    print(f"{'='*60}")

    # Generate 3 responses at each temperature to see variation
    for i in range(3):
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": PROMPT}],
            options={"temperature": temp},
        )
        text = response["message"]["content"].strip()
        print(f"  Run {i+1}: {text[:120]}{'...' if len(text) > 120 else ''}")

print("\n=== What to Notice ===")
print("- At 0.0: responses are identical (deterministic)")
print("- At 0.3: slight variations, same structure")
print("- At 0.7: noticeably different each time")
print("- At 1.0+: may become inconsistent or weird")
print("\nFor manufacturing docs, 0.0-0.3 is your sweet spot.")
```

---

## Exercise 3: Context Window in Practice

```python
# 01-llm-fundamentals/ex3_context_window.py
"""Understand context window limits and their practical impact."""

import ollama

# Let's push the context window and see what happens
short_context = "The weld specification is WPS-201."
long_context = short_context * 500  # Repeat to fill context

print(f"Short context: {len(short_context)} chars (~{len(short_context)//4} tokens)")
print(f"Long context: {len(long_context)} chars (~{len(long_context)//4} tokens)")

# Short context — model remembers everything
print("\n=== Short Context Test ===")
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "user", "content": f"Context: {short_context}\n\nWhat is the weld specification number?"},
    ],
)
print(f"Answer: {response['message']['content'][:200]}")

# Now ask it to work with a LOT of context
# This demonstrates why chunking and RAG matter
print("\n=== Why RAG Matters ===")
print("If you have 500 pages of manufacturing specs, you CAN'T just")
print("paste them all into the prompt. You need to:")
print("  1. Break docs into chunks (Module 08)")
print("  2. Find the RELEVANT chunks (Module 05-06)")
print("  3. Put only those chunks in the prompt (Module 06-07)")
print("  4. Verify the answer is grounded in those chunks (Module 09-10)")
```

---

## Takeaways

After this module, you should understand:

1. **LLMs predict the next token** — everything else is emergent from that
2. **Tokens ≠ words** — technical/manufacturing terminology may use more tokens than you expect
3. **Temperature controls consistency** — for professional task descriptions, keep it low (0.0-0.3)
4. **Context windows are finite** — you can't paste entire manuals into a prompt, which is exactly why RAG exists
5. **The model has no memory between calls** — each API call starts fresh (this matters for evaluation)

## Setting the Stage for Module 02

You understand the theory. Now let's get practical with **running and comparing local models**. Different models have different strengths — some are better at following instructions, some at structured output, some at longer text. Module 02 teaches you to be an informed model shopper.
