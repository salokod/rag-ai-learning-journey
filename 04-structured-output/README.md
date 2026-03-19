# Module 04: Structured Output

## Goal
Get LLMs to return data in exact, machine-parseable formats. This is critical for production systems where downstream code needs to consume LLM output.

---

## Concepts

### Why Structured Output Matters

In a production system, "the LLM wrote some nice text" isn't enough. You need:
- JSON that your application can parse
- Specific fields filled in consistently
- Data types that match your schema
- Outputs that slot into existing systems (databases, forms, APIs)

### Techniques for Structured Output

1. **Prompt-based** — Ask nicely for JSON (unreliable)
2. **Schema in prompt** — Provide the exact JSON schema (better)
3. **Ollama JSON mode** — Force valid JSON output (reliable)
4. **Pydantic parsing** — Validate and parse with Python models (production-grade)

---

## Exercise 1: From Freeform to Structured

```python
# 04-structured-output/ex1_json_output.py
"""Progress from unstructured to structured LLM output."""

import ollama
import json

TASK = "Inspect hydraulic press cylinder seals for wear and leakage"

# Attempt 1: Just ask for JSON (fragile)
print("=== Attempt 1: Ask Nicely ===")
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "user",
            "content": f"Create a task description for: '{TASK}'. Return it as JSON.",
        }
    ],
    options={"temperature": 0.0},
)
print(response["message"]["content"][:300])
print("⚠️  May or may not be valid JSON, may have markdown fences\n")

# Attempt 2: Provide exact schema
print("=== Attempt 2: Provide Schema ===")
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "system",
            "content": """Return task descriptions as JSON matching this exact schema:
{
  "task_id": "string (format: TD-XXXX)",
  "title": "string",
  "department": "string",
  "steps": ["string array of numbered steps"],
  "safety_requirements": ["string array"],
  "quality_checks": ["string array"],
  "estimated_time_minutes": "integer",
  "required_ppe": ["string array"]
}

Return ONLY the JSON, no other text.""",
        },
        {"role": "user", "content": f"Create a task description for: '{TASK}'"},
    ],
    options={"temperature": 0.0},
)
print(response["message"]["content"][:500])

# Attempt 3: Use Ollama's JSON mode (forces valid JSON)
print("\n=== Attempt 3: Ollama JSON Mode ===")
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "system",
            "content": """Return task descriptions as JSON with these fields:
task_id, title, department, steps (array), safety_requirements (array),
quality_checks (array), estimated_time_minutes (int), required_ppe (array).""",
        },
        {"role": "user", "content": f"Create a task description for: '{TASK}'"},
    ],
    format="json",
    options={"temperature": 0.0},
)

result = json.loads(response["message"]["content"])
print(json.dumps(result, indent=2))
print("✓ Guaranteed valid JSON!")
```

---

## Exercise 2: Pydantic Validation

```python
# 04-structured-output/ex2_pydantic_validation.py
"""Use Pydantic to validate and enforce LLM output structure."""

import ollama
import json
from pydantic import BaseModel, Field, field_validator


class TaskDescription(BaseModel):
    """Schema for a manufacturing task description."""

    task_id: str = Field(description="Unique ID in format TD-XXXX")
    title: str = Field(description="Short task title", max_length=100)
    department: str
    steps: list[str] = Field(min_length=3, max_length=7)
    safety_requirements: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(min_length=1)
    estimated_time_minutes: int = Field(ge=1, le=480)
    required_ppe: list[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v):
        if not v.startswith("TD-"):
            raise ValueError("task_id must start with 'TD-'")
        return v

    @field_validator("steps")
    @classmethod
    def steps_start_with_verb(cls, v):
        # Simple check: first word should be capitalized (likely a verb)
        for step in v:
            if step and not step[0].isupper():
                raise ValueError(f"Step should start with an action verb: '{step}'")
        return v


def generate_validated_task(task_input: str) -> TaskDescription:
    """Generate a task description and validate it with Pydantic."""
    schema_json = json.dumps(TaskDescription.model_json_schema(), indent=2)

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": f"Generate task descriptions matching this JSON schema:\n{schema_json}\n"
                "Return ONLY valid JSON.",
            },
            {"role": "user", "content": f"Create a task description for: '{task_input}'"},
        ],
        format="json",
        options={"temperature": 0.0},
    )

    raw = json.loads(response["message"]["content"])

    # Validate with Pydantic — this catches schema violations
    try:
        task = TaskDescription.model_validate(raw)
        print("✓ Validation passed!")
        return task
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        print(f"  Raw output: {json.dumps(raw, indent=2)[:300]}")
        raise


# Test it
tasks = [
    "Calibrate temperature sensors on curing oven",
    "Replace worn conveyor belt rollers",
    "Perform weekly eye wash station inspection",
]

for task_input in tasks:
    print(f"\n{'='*60}")
    print(f"Input: {task_input}")
    print(f"{'='*60}")
    try:
        task = generate_validated_task(task_input)
        print(f"  ID: {task.task_id}")
        print(f"  Title: {task.title}")
        print(f"  Steps: {len(task.steps)}")
        print(f"  PPE: {task.required_ppe}")
        print(f"  Time: {task.estimated_time_minutes} min")
    except Exception:
        print("  (Would retry with adjusted prompt in production)")

print("\n=== Why Pydantic? ===")
print("1. Type safety — catches wrong types before they hit your database")
print("2. Validation — custom rules (ID format, step structure)")
print("3. Documentation — schema IS the documentation")
print("4. Serialization — easy to save/load/transmit")
```

---

## Exercise 3: Structured Output for Evaluation

```python
# 04-structured-output/ex3_structured_for_eval.py
"""Generate structured output that's easy to evaluate programmatically."""

import ollama
import json

# When output is structured, you can evaluate EACH FIELD independently
SYSTEM = """You are a manufacturing task description generator.
Return JSON with these exact fields:
{
  "title": "concise task title",
  "description": "2-3 sentence task description",
  "steps": ["step 1", "step 2", ...],
  "hazards": ["identified hazards"],
  "ppe_required": ["required PPE items"],
  "tools_needed": ["tools and equipment"],
  "acceptance_criteria": "how to know the task is done correctly",
  "estimated_minutes": integer
}"""

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Task: Grind down weld spatter on steel frame"},
    ],
    format="json",
    options={"temperature": 0.0},
)

task = json.loads(response["message"]["content"])
print("Generated task:")
print(json.dumps(task, indent=2))

# Now we can evaluate EACH field programmatically
print("\n=== Programmatic Evaluation (preview of Module 09) ===")
checks = {
    "has_title": bool(task.get("title")),
    "has_description": bool(task.get("description")),
    "has_steps": len(task.get("steps", [])) >= 3,
    "has_hazards": len(task.get("hazards", [])) >= 1,
    "has_ppe": len(task.get("ppe_required", [])) >= 1,
    "has_tools": len(task.get("tools_needed", [])) >= 1,
    "has_acceptance_criteria": bool(task.get("acceptance_criteria")),
    "reasonable_time": 1 <= task.get("estimated_minutes", 0) <= 480,
    "description_length": 20 <= len(task.get("description", "").split()) <= 100,
}

total = len(checks)
passed = sum(checks.values())
print(f"\nScore: {passed}/{total} checks passed ({100*passed/total:.0f}%)")
for check, result in checks.items():
    status = "✓" if result else "✗"
    print(f"  {status} {check}")

print("\nThis is the START of evaluation. Modules 09-13 make this much more sophisticated.")
```

---

## Takeaways

1. **Always use JSON mode** (`format="json"`) when you need structured output — don't rely on the model "being nice"
2. **Pydantic models** give you type safety, validation, and documentation in one place
3. **Structured output enables structured evaluation** — you can check each field independently
4. **Schema-first design** — define your output schema before writing the prompt
5. **Retry logic** is essential in production — LLMs occasionally produce invalid output even with JSON mode

## Setting the Stage for Module 05

You can generate structured task descriptions, but they're based entirely on the LLM's training data. What about YOUR company's specific procedures, equipment, and standards? Module 05 introduces **embeddings and vector stores** — the technology that lets you search your own documents semantically. This is the foundation of RAG.
