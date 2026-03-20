# 01-llm-fundamentals/ex8_combined.py
"""Experiment with LLM parameters for manufacturing task descriptions."""
import ollama


def generate(prompt, temperature=0.0, top_p=1.0):
    """Generate a response with specific settings."""
    r = ollama.chat(
        model="gemma3:12b",
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
    print(f"  {result[:1500]}...")
    print()


# --- Experiment 2: Brainstorming mode ---
print("=== Experiment 2: Brainstorming task variations ===")
print("(temperature=0.9 -- for exploring different phrasings)\n")

for i in range(3):
    result = generate(task_prompt, temperature=0.9)
    print(f"Variation {i+1}:")
    print(f"  {result[:1500]}...")
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
    print(f"  {result[:1200]}...")
    print()

print("Notice how more specific context produces more specific output.")
print("This is exactly why RAG works -- better context = better answers.")