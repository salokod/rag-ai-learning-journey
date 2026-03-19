# Module 03: Prompt Engineering

## Goal
Master systematic prompt engineering techniques. By the end, you'll have a prompt suite that can generate manufacturing task descriptions matching a specific style and format.

---

## Concepts

### Prompt Engineering Is Not Magic

It's not about "one weird trick." It's a systematic process:

1. **Define what "good" looks like** — collect examples of the output you want
2. **Structure your prompt** — use proven techniques (zero-shot, few-shot, chain-of-thought)
3. **Iterate and measure** — change one thing at a time, evaluate the result
4. **Lock it down** — once it works, version control it like code

### The Anatomy of a Prompt

```
┌─────────────────────────────────────────┐
│ SYSTEM PROMPT                           │
│ - Role definition                       │
│ - Constraints and rules                 │
│ - Output format specification           │
├─────────────────────────────────────────┤
│ FEW-SHOT EXAMPLES (optional)            │
│ - Input → Expected output pairs         │
│ - Shows the model what "good" looks like│
├─────────────────────────────────────────┤
│ USER INPUT                              │
│ - The specific task/question            │
│ - Any context or reference material     │
└─────────────────────────────────────────┘
```

### Key Techniques

| Technique | When to Use | Example |
|-----------|-------------|---------|
| **Zero-shot** | Simple tasks, model already knows how | "Summarize this paragraph" |
| **Few-shot** | Need specific format/style | Provide 2-3 examples first |
| **Chain-of-thought** | Complex reasoning needed | "Think through this step by step" |
| **Role prompting** | Domain-specific language | "You are a manufacturing engineer" |
| **Constrained output** | Exact format needed | "Respond in exactly this JSON format" |

---

## Exercise 1: Zero-Shot vs Few-Shot

```python
# 03-prompt-engineering/ex1_zero_vs_few_shot.py
"""Compare zero-shot and few-shot prompting for task descriptions."""

import ollama

# === ZERO-SHOT: Just tell it what you want ===
zero_shot_prompt = """Write a manufacturing task description for:
"Inspect incoming raw steel plates for surface defects"

The description should be professional, specific, and actionable."""

print("=== ZERO-SHOT ===")
response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": zero_shot_prompt}],
    options={"temperature": 0.1},
)
print(response["message"]["content"])

# === FEW-SHOT: Show it examples first ===
few_shot_prompt = """You write manufacturing task descriptions. Here are examples of the correct format:

EXAMPLE 1:
Task: Verify torque on fastener assembly
Description: Using a calibrated torque wrench (±2% accuracy), verify all fasteners on Assembly #4200 meet specification MT-302 requirements. Check each fastener in sequence per the torque map diagram. Record actual torque values on Form QC-110. Any fastener outside the 25-30 Nm range must be flagged and reported to the shift supervisor before proceeding.

EXAMPLE 2:
Task: Clean CNC machine coolant reservoir
Description: Drain coolant reservoir completely using the designated waste container (yellow, labeled "Used Coolant"). Remove debris from the screen filter and inspect for damage. Flush reservoir with clean water, then refill with Type III coolant to the MAX line. Log the coolant change on the machine maintenance card and initial the daily checklist.

Now write a description in the same format for:
Task: Inspect incoming raw steel plates for surface defects"""

print("\n\n=== FEW-SHOT ===")
response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": few_shot_prompt}],
    options={"temperature": 0.1},
)
print(response["message"]["content"])

print("\n\n=== COMPARISON ===")
print("Notice how few-shot output:")
print("  - Matches the format of the examples")
print("  - Uses similar level of detail")
print("  - Includes specific references (forms, specs, tools)")
print("  - Has a similar tone and sentence structure")
print("\nThis is the foundation of style matching — and we'll measure it in Module 09.")
```

---

## Exercise 2: System Prompts and Role Definition

```python
# 03-prompt-engineering/ex2_system_prompts.py
"""Learn how system prompts shape model behavior."""

import ollama

task = "Write a task description for: 'Replace worn conveyor belt rollers'"

# Different system prompts, same task
system_prompts = {
    "no_system": None,
    "basic_role": "You are a manufacturing technical writer.",
    "detailed_role": """You are a senior manufacturing technical writer at an ISO 9001-certified facility.
You write task descriptions that:
- Start with the action verb
- Include specific tools, specifications, and forms by reference number
- Include safety requirements (PPE, lockout/tagout) when applicable
- Are written at an 8th-grade reading level for operator accessibility
- Are 50-100 words long
- End with the quality check or sign-off requirement""",
    "chain_of_thought": """You are a manufacturing technical writer. Before writing,
think through:
1. What tools/equipment does the operator need?
2. What safety precautions apply?
3. What are the quality/acceptance criteria?
4. What documentation is required?

Then write a clear, professional task description incorporating your analysis.""",
}

for name, system_prompt in system_prompts.items():
    print(f"\n{'='*60}")
    print(f"System Prompt: {name}")
    print(f"{'='*60}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": task})

    response = ollama.chat(
        model="llama3.1:8b",
        messages=messages,
        options={"temperature": 0.1},
    )
    print(response["message"]["content"][:400])

print("\n=== Key Insight ===")
print("The 'detailed_role' system prompt with specific constraints produces")
print("the most consistent, professional output. This is what you'd use in production.")
```

---

## Exercise 3: Building a Reusable Prompt Template

```python
# 03-prompt-engineering/ex3_prompt_template.py
"""Build a production-quality prompt template for manufacturing tasks."""

import ollama
from string import Template

# Your production prompt template — version control this!
TASK_DESCRIPTION_SYSTEM = """You are a technical writer for a manufacturing facility.
You produce task descriptions following these exact rules:

FORMAT:
- First line: Task title in CAPS
- Second line: blank
- Body: 3-5 numbered steps
- Each step starts with an action verb
- Include tool/equipment references in parentheses where applicable
- Include specification/form references where applicable
- Final step must be a documentation or quality verification step

STYLE:
- Active voice ("Inspect the..." not "The part should be inspected...")
- 8th-grade reading level
- No jargon without definition on first use
- Specific and measurable where possible

SAFETY:
- If the task involves machinery, step 1 must address lockout/tagout or safety
- Always reference required PPE"""

TASK_DESCRIPTION_USER = Template("""Write a task description for the following:

Task: $task_name
Department: $department
Equipment: $equipment
Relevant specifications: $specifications

Write ONLY the task description, no other commentary.""")


def generate_task_description(task_name, department, equipment, specifications):
    """Generate a standardized task description."""
    user_prompt = TASK_DESCRIPTION_USER.substitute(
        task_name=task_name,
        department=department,
        equipment=equipment,
        specifications=specifications,
    )

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": TASK_DESCRIPTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1, "repeat_penalty": 1.2},
    )
    return response["message"]["content"]


# Test with several different tasks
test_tasks = [
    {
        "task_name": "Inspect welded joints on Frame Assembly A",
        "department": "Quality Control",
        "equipment": "Visual inspection kit, ultrasonic thickness gauge",
        "specifications": "AWS D1.1, Internal spec WPS-201",
    },
    {
        "task_name": "Set up CNC lathe for shaft machining",
        "department": "Machining",
        "equipment": "Haas ST-20 CNC lathe, tool setter",
        "specifications": "Drawing #SH-4402-Rev.B, tolerance ±0.005\"",
    },
    {
        "task_name": "Perform daily forklift inspection",
        "department": "Warehouse",
        "equipment": "Toyota 8FGU25 forklift",
        "specifications": "OSHA 1910.178, Company SOP-FL-001",
    },
]

for task in test_tasks:
    print(f"\n{'='*60}")
    print(f"Task: {task['task_name']}")
    print(f"{'='*60}")
    description = generate_task_description(**task)
    print(description)

print("\n=== What You Built ===")
print("A reusable, version-controlled prompt template that produces")
print("consistent task descriptions. In Module 06, we'll feed it")
print("reference documents via RAG. In Module 09, we'll evaluate quality.")
```

---

## Exercise 4: Prompt Iteration Tracking

```python
# 03-prompt-engineering/ex4_prompt_versioning.py
"""Track prompt iterations to know which version performs best."""

import ollama
import json
from datetime import datetime

# Prompt versions — track changes over time
PROMPT_VERSIONS = {
    "v1": {
        "system": "You write manufacturing task descriptions.",
        "notes": "Minimal system prompt, no constraints",
    },
    "v2": {
        "system": """You write manufacturing task descriptions.
Keep them under 100 words. Use active voice. Include safety notes.""",
        "notes": "Added basic constraints",
    },
    "v3": {
        "system": """You are a technical writer for an ISO 9001 facility.
Write task descriptions with numbered steps (3-5 steps).
Start each step with an action verb.
Include tool references in parentheses.
Final step must be documentation/verification.
Active voice. Under 100 words.""",
        "notes": "Detailed format, style, and structure constraints",
    },
}

test_input = "Write a task description for: Calibrate digital pressure gauge"

results = []

for version, config in PROMPT_VERSIONS.items():
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": test_input},
        ],
        options={"temperature": 0.0},
    )

    output = response["message"]["content"]
    result = {
        "version": version,
        "notes": config["notes"],
        "output": output,
        "word_count": len(output.split()),
        "has_numbered_steps": any(f"{i}." in output for i in range(1, 6)),
        "starts_with_verb": output.strip()[0:20],  # Manual check
        "timestamp": datetime.now().isoformat(),
    }
    results.append(result)

    print(f"\n{'='*60}")
    print(f"{version}: {config['notes']}")
    print(f"Words: {result['word_count']} | Numbered steps: {result['has_numbered_steps']}")
    print(f"{'='*60}")
    print(output[:300])

# Save results for comparison
with open("03-prompt-engineering/prompt_iterations.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\n=== Prompt Versioning Lesson ===")
print("Saved results to prompt_iterations.json")
print("As you iterate prompts, ALWAYS:")
print("  1. Save the prompt version")
print("  2. Save the output")
print("  3. Track what changed and why")
print("  4. Evaluate systematically (not just 'looks good')")
print("\nThis manual tracking is exactly what Langfuse automates (Module 12).")
```

---

## Takeaways

1. **Few-shot prompting** is your most powerful tool for style matching — show the model what you want
2. **System prompts** define the model's behavior constraints — be specific and detailed
3. **Prompt templates** should be version-controlled like code — they ARE code
4. **Iteration must be tracked** — random tweaking without measurement is guessing, not engineering
5. **Your manufacturing prompt template** is a living document that will improve through the journey

## Setting the Stage for Module 04

Your prompts generate good text, but it comes back as unstructured strings. For production systems, you need the LLM to output **structured data** — JSON, specific schemas, exactly the fields you need. Module 04 teaches you to constrain LLM output so it's machine-parseable, not just human-readable.
