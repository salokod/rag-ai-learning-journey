# Module 04: Structured Output

## What You'll Learn

How to make an LLM return data in a format your code can actually use -- JSON with specific fields, validated by Python. By the end, you'll have a bullet-proof pipeline that catches the LLM when it makes things up.

**Time:** ~60 minutes of hands-on work

**Prerequisites:** Module 03 complete (you have Ollama running and understand prompt templates)

---

## Part 1: The Problem -- "Just Give Me JSON"

Let's start with the most obvious approach. Open a Python shell:

```bash
python3
```

Now ask the model for JSON the naive way:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the library requires something
)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "Describe the task 'Inspect hydraulic press seals'. Return as JSON."}],
)
print(response.choices[0].message.content)
```

Run that. What do you get back?

Look carefully at the output. You'll probably see something like:

```
Here's the task as JSON:

```json
{
  "task": "Inspect hydraulic press seals",
  ...
}
```​

See the problem? The model wrapped it in markdown code fences. It added a sentence before the JSON. Try parsing that:

```python
import json
json.loads(response.choices[0].message.content)
```

Boom -- `json.JSONDecodeError`. Your code can't use this. Run it a few more times and you might get different wrapping each time. Sometimes markdown fences, sometimes a preamble sentence, sometimes bare JSON if you're lucky.

In a production system -- say, a system that generates task descriptions and stores them in your manufacturing database -- "sometimes works" is not acceptable.

---

## Part 2: Ollama's JSON Mode

Ollama has a built-in fix. Let's try it:

```python
response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "Describe the task 'Inspect hydraulic press seals' as JSON."}],
    response_format={"type": "json_object"},
)
print(response.choices[0].message.content)
```

Notice the difference? No markdown fences. No preamble. Just raw JSON. Let's verify it parses:

```python
result = json.loads(response.choices[0].message.content)
print(type(result))
```

You should see `<class 'dict'>`. Every time, guaranteed.

But look at the actual fields in the JSON:

```python
print(json.dumps(result, indent=2))
```

What fields did you get? Probably something random -- maybe `"task"`, maybe `"name"`, maybe `"description"`. The model picked whatever it felt like. That's still a problem. If your database expects a `task_id` column and the LLM returns `id`, your code breaks.

---

## Part 3: Telling the Model What Fields You Want

Let's combine `format="json"` with a schema in the prompt. Try this:

```python
response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": """Return JSON with exactly these fields:
- task_id: string in format "TD-XXXX"
- title: short task title
- steps: array of action steps
- safety: array of safety requirements""",
        },
        {"role": "user", "content": "Task: Inspect hydraulic press cylinder seals for wear and leakage"},
    ],
    response_format={"type": "json_object"},
)

result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2))
```

Now check -- did you get the exact field names you asked for?

```python
print(list(result.keys()))
```

Much better. The fields match what we asked for. But here's a question: what's the type of `result["steps"]`?

```python
print(type(result["steps"]))
print(type(result["task_id"]))
```

It's a list of strings and a string. But what if the model returned `"steps": "Step 1, Step 2, Step 3"` as a single string instead of an array? Or what if `task_id` came back as `"TASK-1234"` instead of `"TD-1234"`?

Your code would crash somewhere downstream. We need validation.

---

## Part 4: Enter Pydantic -- Your Safety Net

Pydantic is a Python library for data validation. Let's start small. In a new file or fresh shell:

```python
from pydantic import BaseModel
```

That's it for now. Let's build our first tiny model:

```python
class SimpleTask(BaseModel):
    title: str
    steps: list[str]
    time_minutes: int
```

Three fields. That's our schema. Now let's test it with some data:

```python
task = SimpleTask(title="Check seals", steps=["Step 1", "Step 2"], time_minutes=30)
print(task)
```

Works fine. Now let's see what happens with bad data:

```python
task = SimpleTask(title="Check seals", steps="not a list", time_minutes=30)
```

What happened? Pydantic actually coerced `"not a list"` into a list with one element. Interesting. Try something it really can't fix:

```python
task = SimpleTask(title="Check seals", steps=["Step 1"], time_minutes="not a number")
```

`ValidationError`. That's exactly what we want. Pydantic catches problems before they reach your database.

---

## Part 5: Adding Validation Rules

Let's make our model smarter, one rule at a time. Start fresh:

```python
from pydantic import BaseModel, Field

class TaskDescription(BaseModel):
    task_id: str
    title: str
    steps: list[str]
```

Simple. Now let's add a rule -- task_id should follow a format:

```python
from pydantic import BaseModel, Field, field_validator

class TaskDescription(BaseModel):
    task_id: str
    title: str
    steps: list[str]

    @field_validator("task_id")
    @classmethod
    def check_task_id_format(cls, v):
        if not v.startswith("TD-"):
            raise ValueError("task_id must start with 'TD-'")
        return v
```

Test it:

```python
TaskDescription(task_id="TD-0001", title="Test", steps=["Step 1"])
```

Works. Now:

```python
TaskDescription(task_id="TASK-0001", title="Test", steps=["Step 1"])
```

`ValidationError: task_id must start with 'TD-'`. The LLM often invents its own ID formats. This catches that.

Let's add more rules. Think about what matters for manufacturing task descriptions:

```python
class TaskDescription(BaseModel):
    task_id: str
    title: str = Field(max_length=100)
    steps: list[str] = Field(min_length=2, max_length=10)
    safety_requirements: list[str] = Field(default_factory=list)
    estimated_time_minutes: int = Field(ge=1, le=480)

    @field_validator("task_id")
    @classmethod
    def check_task_id_format(cls, v):
        if not v.startswith("TD-"):
            raise ValueError("task_id must start with 'TD-'")
        return v
```

Notice what each rule does:

- `max_length=100` on title: no essay-length titles
- `min_length=2` on steps: a task with one step is suspicious
- `max_length=10` on steps: a task with 20 steps should be split up
- `ge=1, le=480` on time: at least 1 minute, at most 8 hours
- `default_factory=list` on safety: if the LLM forgets safety, we get an empty list instead of an error

Try a few test cases:

```python
# Too many steps?
TaskDescription(
    task_id="TD-0001",
    title="Test",
    steps=["s"] * 15,
    estimated_time_minutes=30,
)
```

```python
# Negative time?
TaskDescription(
    task_id="TD-0001",
    title="Test",
    steps=["Step 1", "Step 2"],
    estimated_time_minutes=-5,
)
```

Each validation error is a GOOD thing -- it means you caught the LLM before it corrupted your data.

---

## Part 6: Connecting Pydantic to the LLM

Now the real magic. Let's feed the Pydantic schema to the LLM so it knows exactly what we expect:

```python
import json

print(json.dumps(TaskDescription.model_json_schema(), indent=2))
```

Look at that output. Pydantic auto-generates a JSON schema from your model. The LLM can read this and know exactly what fields and types to produce.

Let's wire it all together:

```python
from openai import OpenAI
import json
from pydantic import BaseModel, Field, field_validator

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


class TaskDescription(BaseModel):
    task_id: str
    title: str = Field(max_length=100)
    steps: list[str] = Field(min_length=2, max_length=10)
    safety_requirements: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    estimated_time_minutes: int = Field(ge=1, le=480)
    required_ppe: list[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def check_task_id_format(cls, v):
        if not v.startswith("TD-"):
            raise ValueError("task_id must start with 'TD-'")
        return v


schema = json.dumps(TaskDescription.model_json_schema(), indent=2)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": f"Generate manufacturing task descriptions as JSON matching this schema:\n{schema}",
        },
        {"role": "user", "content": "Task: Calibrate temperature sensors on curing oven"},
    ],
    response_format={"type": "json_object"},
)

raw = json.loads(response.choices[0].message.content)
print("Raw from LLM:")
print(json.dumps(raw, indent=2))
```

Now validate it:

```python
try:
    task = TaskDescription.model_validate(raw)
    print(f"\nValidation passed!")
    print(f"  ID:    {task.task_id}")
    print(f"  Title: {task.title}")
    print(f"  Steps: {len(task.steps)}")
    print(f"  PPE:   {task.required_ppe}")
    print(f"  Time:  {task.estimated_time_minutes} min")
except Exception as e:
    print(f"\nValidation FAILED: {e}")
    print("The LLM returned something that doesn't match our rules.")
```

Did it pass? If the task_id didn't start with "TD-", you'll see the validation error. That's exactly the point -- you caught it before it hit your database.

---

## Part 7: What Happens When Validation Fails

Let's deliberately break things to understand the safety net. Create some fake "LLM output" with problems:

```python
bad_outputs = [
    {"task_id": "TASK-99", "title": "Test", "steps": ["one step"], "estimated_time_minutes": 30},
    {"task_id": "TD-0001", "title": "T" * 200, "steps": ["a", "b"], "estimated_time_minutes": 30},
    {"task_id": "TD-0001", "title": "Test", "steps": ["a", "b"], "estimated_time_minutes": 999},
]

for i, bad in enumerate(bad_outputs):
    print(f"\nTest {i + 1}: ", end="")
    try:
        TaskDescription.model_validate(bad)
        print("Passed (unexpected!)")
    except Exception as e:
        print(f"Caught: {e}")
```

What did each one catch?

1. `"TASK-99"` -- wrong ID format (not "TD-")
2. Title with 200 characters -- exceeds max_length
3. 999 minutes -- exceeds the 480-minute max

In production, when validation fails, you'd retry the LLM call (maybe with a more explicit prompt) or log it for human review. The key insight: **validation turns silent data corruption into loud, catchable errors.**

---

## Part 8: Putting It All Together

Time for the full exercise. Let's combine everything -- your prompt engineering from Module 03, JSON mode, and Pydantic validation.

Save this as `04-structured-output/full_pipeline.py`:

```python
"""Full structured output pipeline: prompt + JSON mode + Pydantic validation."""

from openai import OpenAI
import json
from pydantic import BaseModel, Field, field_validator

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the library requires something
)


class TaskDescription(BaseModel):
    """Manufacturing task description -- validated schema."""

    task_id: str
    title: str = Field(max_length=100)
    department: str
    steps: list[str] = Field(min_length=2, max_length=10)
    safety_requirements: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(min_length=1)
    estimated_time_minutes: int = Field(ge=1, le=480)
    required_ppe: list[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def check_task_id_format(cls, v):
        if not v.startswith("TD-"):
            raise ValueError("task_id must start with 'TD-'")
        return v

    @field_validator("steps")
    @classmethod
    def steps_start_with_verb(cls, v):
        for step in v:
            if step and not step[0].isupper():
                raise ValueError(f"Each step should start with a capital letter (action verb): '{step}'")
        return v


def generate_task(task_input: str, max_retries: int = 2) -> TaskDescription | None:
    """Generate a validated task description with retry logic."""
    schema = json.dumps(TaskDescription.model_json_schema(), indent=2)

    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model="gemma3:12b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a manufacturing task description generator.\n"
                        f"Return JSON matching this schema:\n{schema}\n\n"
                        "Rules:\n"
                        "- task_id must start with 'TD-' followed by 4 digits\n"
                        "- Each step must start with an action verb (capitalized)\n"
                        "- Include relevant safety requirements and PPE\n"
                        "- Quality checks should be measurable when possible"
                    ),
                },
                {"role": "user", "content": f"Create a task description for: {task_input}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw = json.loads(response.choices[0].message.content)

        try:
            task = TaskDescription.model_validate(raw)
            if attempt > 0:
                print(f"  (passed on retry {attempt})")
            return task
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed validation: {e}")
            if attempt == max_retries:
                print("  All retries exhausted.")
                return None

    return None


# --- Try it with several manufacturing tasks ---

tasks = [
    "Inspect hydraulic press cylinder seals for wear and leakage",
    "Calibrate temperature sensors on curing oven",
    "Replace worn conveyor belt rollers",
    "Perform weekly eye wash station inspection",
]

for task_input in tasks:
    print(f"\n{'=' * 60}")
    print(f"Input: {task_input}")
    print(f"{'=' * 60}")

    task = generate_task(task_input)

    if task:
        print(f"  ID:      {task.task_id}")
        print(f"  Title:   {task.title}")
        print(f"  Dept:    {task.department}")
        print(f"  Steps:   {len(task.steps)}")
        for s in task.steps:
            print(f"           - {s}")
        print(f"  Safety:  {task.safety_requirements}")
        print(f"  PPE:     {task.required_ppe}")
        print(f"  Checks:  {task.quality_checks}")
        print(f"  Time:    {task.estimated_time_minutes} min")
    else:
        print("  FAILED -- would need human review")
```

Run it:

```bash
cd 04-structured-output
python full_pipeline.py
```

Watch what happens. For each task, you should see validated, consistently-structured output. If any validation fails, the retry logic kicks in.

---

## Part 9: Quick Evaluation Preview

One last thing. Because the output is structured, you can evaluate each field programmatically. This is a taste of what's coming in Modules 09-13:

```python
def quick_eval(task: TaskDescription) -> dict:
    """Basic quality checks on a generated task."""
    checks = {
        "has_3_plus_steps": len(task.steps) >= 3,
        "has_safety": len(task.safety_requirements) >= 1,
        "has_ppe": len(task.required_ppe) >= 1,
        "has_quality_checks": len(task.quality_checks) >= 1,
        "reasonable_time": 5 <= task.estimated_time_minutes <= 240,
        "title_not_too_long": len(task.title) <= 80,
    }
    passed = sum(checks.values())
    total = len(checks)
    return {"score": f"{passed}/{total}", "checks": checks}
```

Try it on one of your generated tasks:

```python
task = generate_task("Grind down weld spatter on steel frame")
if task:
    result = quick_eval(task)
    print(f"Score: {result['score']}")
    for check, passed in result["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {check}")
```

Structured output makes structured evaluation possible. When everything is free-form text, evaluation is much harder.

---

## What You Now Know

- **`format="json"`** forces Ollama to return valid JSON every time
- **Schema in the prompt** tells the LLM which fields to include
- **Pydantic models** validate the output -- catching type mismatches, missing fields, and format violations
- **Validation failures are features, not bugs** -- they prevent bad data from reaching your systems
- **Retry logic** handles the occasional LLM hiccup gracefully
- **Structured output enables structured evaluation** -- you can check each field independently

## Up Next: Module 05

You can now generate structured task descriptions, but they're based entirely on the LLM's training data. What about YOUR company's specific procedures, equipment specs, and safety standards? Module 05 introduces **embeddings and vector stores** -- the technology that lets you search your own documents by meaning. That's the foundation of RAG.
