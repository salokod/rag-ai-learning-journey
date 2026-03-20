# Module 03: Prompt Engineering

## Goal
Learn to control what an LLM produces -- systematically, not by luck. By the end of this module, you'll have a reusable prompt template that generates manufacturing task descriptions in your style, your format, every time. This is the most important module for your project.

---

## Part 1: Start Simple -- Zero-Shot Prompting

Zero-shot means you just ask for what you want. No examples, no special setup. Let's see what happens.

### Step 1: Just ask

Create a file and run it:

```python
# 03-prompt-engineering/step01_zero_shot.py
import ollama

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "user", "content": "Write a task description for: Inspect incoming raw steel plates for surface defects"}
    ],
    options={"temperature": 0.1},
)

print(response["message"]["content"])
```

```bash
python 03-prompt-engineering/step01_zero_shot.py
```

Read the output. It's... OK, right? It probably mentions visual inspection, maybe some defect types. But ask yourself:

- Does it match the format your facility actually uses?
- Is it the right length?
- Does it mention specific tools, forms, or spec numbers?
- Would you hand this to a new operator on the floor?

Probably not. The model is guessing at what a "task description" should look like. It has no idea what *your* task descriptions look like.

Let's fix that, one piece at a time.

---

## Part 2: System Prompts -- Giving the Model a Job

### Step 2: Add a role

Instead of talking to a generic AI, let's tell it who it is:

```python
# 03-prompt-engineering/step02_system_prompt.py
import ollama

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "You are a manufacturing technical writer."},
        {"role": "user", "content": "Write a task description for: Inspect incoming raw steel plates for surface defects"},
    ],
    options={"temperature": 0.1},
)

print(response["message"]["content"])
```

Run it. Compare to Step 1.

See the difference? Even that one line -- "You are a manufacturing technical writer" -- shifts the tone. The language gets more professional. It might start including things like PPE or documentation steps that were missing before.

But it's still pretty vague about the *format*. Let's get more specific.

### Step 3: Add constraints

```python
# 03-prompt-engineering/step03_detailed_system.py
import ollama

system_prompt = """You are a senior technical writer at an ISO 9001-certified manufacturing facility.
You write task descriptions that:
- Start with an action verb
- Include specific tools and equipment in parentheses
- Reference applicable specifications or form numbers
- Include safety requirements (PPE, lockout/tagout) when applicable
- Are written at an 8th-grade reading level
- Are 50-100 words long
- End with a quality check or sign-off step"""

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Write a task description for: Inspect incoming raw steel plates for surface defects"},
    ],
    options={"temperature": 0.1},
)

print(response["message"]["content"])
```

Run it. NOW look at the difference.

This is a big jump. The model is following your constraints -- numbered steps, action verbs, tool references. It's starting to look like a real work instruction.

Notice what happened: we didn't change the question at all. We only changed the *framing*. The system prompt is like a job description for the AI -- the more specific you are about the job, the better it performs.

### Try this: Change one constraint

Go back to the system prompt and change "50-100 words" to "exactly 3 steps." Run it again. See how it adapts? Each constraint you add or change steers the output. This is the core of prompt engineering -- not magic, just clear instructions.

---

## Part 3: Few-Shot Prompting -- Show, Don't Tell

This is your most powerful technique for manufacturing work. Instead of *describing* the format you want, you *show* the model examples.

### Step 4: Give it examples

```python
# 03-prompt-engineering/step04_few_shot.py
import ollama

few_shot_prompt = """You write manufacturing task descriptions. Here are examples of the correct format:

EXAMPLE 1:
Task: Verify torque on fastener assembly
Description: Using a calibrated torque wrench (accuracy +/-2%), verify all fasteners on Assembly #4200 meet specification MT-302 requirements. Check each fastener in sequence per the torque map diagram. Record actual torque values on Form QC-110. Any fastener outside the 25-30 Nm range must be flagged and reported to the shift supervisor before proceeding.

EXAMPLE 2:
Task: Clean CNC machine coolant reservoir
Description: Drain coolant reservoir completely using the designated waste container (yellow, labeled "Used Coolant"). Remove debris from the screen filter and inspect for damage. Flush reservoir with clean water, then refill with Type III coolant to the MAX line. Log the coolant change on the machine maintenance card and initial the daily checklist.

Now write a description in the same format for:
Task: Inspect incoming raw steel plates for surface defects"""

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": few_shot_prompt}],
    options={"temperature": 0.1},
)

print(response["message"]["content"])
```

Run it. Compare this to every previous version.

THIS is the big moment. Look at what happened:
- The format matches your examples (Task/Description structure)
- The level of detail is similar (specific tools, form numbers, thresholds)
- The tone matches (professional but accessible)
- It even invented plausible form numbers and spec references in the same style

The model learned your format from just two examples. This is few-shot prompting, and it's the foundation of the entire manufacturing task description project.

### Step 5: What happens with only one example?

Try removing EXAMPLE 2 from the prompt and running again. What changes?

With one example, the model has less to pattern-match on. Two examples is usually the minimum for reliable style matching. Three is even better -- but there's a point of diminishing returns around 3-5 examples.

### Step 6: Combine system prompt + few-shot

Now let's combine what we've learned:

```python
# 03-prompt-engineering/step06_combined.py
import ollama

system_prompt = """You are a senior technical writer at an ISO 9001-certified manufacturing facility.
You write task descriptions following these rules:
- Active voice, 8th-grade reading level
- Include specific tools in parentheses
- Reference form numbers and specifications
- Include safety requirements when applicable
- 50-100 words per description
- End with a documentation or quality verification step"""

user_prompt = """Here are examples of correct task descriptions:

Task: Verify torque on fastener assembly
Description: Using a calibrated torque wrench (accuracy +/-2%), verify all fasteners on Assembly #4200 meet specification MT-302 requirements. Check each fastener in sequence per the torque map diagram. Record actual torque values on Form QC-110. Any fastener outside the 25-30 Nm range must be flagged and reported to the shift supervisor before proceeding.

Task: Clean CNC machine coolant reservoir
Description: Drain coolant reservoir completely using the designated waste container (yellow, labeled "Used Coolant"). Remove debris from the screen filter and inspect for damage. Flush reservoir with clean water, then refill with Type III coolant to the MAX line. Log the coolant change on the machine maintenance card and initial the daily checklist.

Now write a description for:
Task: Replace worn conveyor belt rollers"""

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    options={"temperature": 0.1, "repeat_penalty": 1.2},
)

print(response["message"]["content"])
```

Run it. This is the formula: **system prompt (rules) + few-shot examples (style) + your request**. This combination is what you'll use in production.

---

## Part 4: Chain-of-Thought -- Making the Model Think First

Sometimes you want the model to reason before writing. This is especially useful for complex tasks where it needs to consider safety, tools, and documentation requirements.

### Step 7: Think, then write

```python
# 03-prompt-engineering/step07_chain_of_thought.py
import ollama

system_prompt = """You are a manufacturing technical writer. Before writing a task description,
think through these questions:
1. What tools or equipment does the operator need?
2. What safety precautions apply (PPE, lockout/tagout)?
3. What are the quality or acceptance criteria?
4. What documentation must be completed?

Show your thinking, then write the final task description."""

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Write a task description for: Set up CNC lathe for shaft machining run"},
    ],
    options={"temperature": 0.1},
)

print(response["message"]["content"])
```

Run it. Notice how the output has two parts: the model's reasoning (answering those four questions) and then the actual task description.

This is **chain-of-thought prompting**. The reasoning step forces the model to consider all the important factors before writing. The result is usually more thorough and complete.

**When to use it:** For complex tasks where the model might miss important details. For simple tasks, it's overkill -- just use the system prompt + few-shot approach.

**In production**, you'd probably hide the thinking section and only show the final description to the user. But during development, seeing the model's reasoning helps you debug when the output is wrong.

---

## Part 5: Prompt Versioning -- Track What Works

Here's something your boss will care about. When someone asks "can we trust this AI system?", you need to show the progression. You need to show that you tested different prompts, measured the results, and chose the best one.

Let's build a simple A/B testing setup.

### Step 8: Compare prompt versions

```python
# 03-prompt-engineering/step08_prompt_versions.py
import ollama
import json
from datetime import datetime

# Define prompt versions -- each one is an iteration
PROMPT_VERSIONS = {
    "v1": {
        "system": "You write manufacturing task descriptions.",
        "notes": "Minimal -- just the bare instruction",
    },
    "v2": {
        "system": """You write manufacturing task descriptions.
Keep them under 100 words. Use active voice. Include safety notes.""",
        "notes": "Added basic constraints: length, voice, safety",
    },
    "v3": {
        "system": """You are a technical writer for an ISO 9001 facility.
Write task descriptions with numbered steps (3-5 steps).
Start each step with an action verb.
Include tool references in parentheses.
Final step must be documentation or verification.
Active voice. 50-100 words.""",
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
        "timestamp": datetime.now().isoformat(),
    }
    results.append(result)

    print(f"\n{'=' * 60}")
    print(f"{version}: {config['notes']}")
    print(f"Words: {result['word_count']} | Numbered steps: {result['has_numbered_steps']}")
    print(f"{'=' * 60}")
    print(output[:400])

# Save results
with open("03-prompt-engineering/prompt_iterations.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print("Results saved to prompt_iterations.json")
print(f"{'=' * 60}")
print("\nLook at the progression:")
print("  v1 -> Generic, unstructured, inconsistent length")
print("  v2 -> Better, but format is still unpredictable")
print("  v3 -> Numbered steps, action verbs, consistent structure")
print("\nEach version is an improvement because we changed ONE thing")
print("and checked the result. This is prompt engineering as a process.")
```

Run it:

```bash
python 03-prompt-engineering/step08_prompt_versions.py
```

Open the `prompt_iterations.json` file and look at the data. You've got a record of what you tried, what changed, and what the model produced. This is exactly the kind of evidence you'd show when someone questions the AI system.

### Try this: Create a v4

Add a v4 to the `PROMPT_VERSIONS` dictionary. Try adding few-shot examples to the system prompt, or add a constraint about safety being mentioned in step 1. Run it again. Does v4 beat v3? Save and compare.

This manual tracking is what observability tools like Langfuse (Module 12) automate. But understanding the manual process first makes the automation make sense.

---

## Part 6: Building Your Reusable Prompt Template

Everything you've learned so far comes together here. Let's build a production-quality prompt template function that you'll actually use in your capstone project.

### Step 9: The template

```python
# 03-prompt-engineering/step09_template.py
import ollama
from string import Template

# --- Your production prompt template ---
# Version control this. Treat it like code, because it IS code.

SYSTEM_PROMPT = """You are a technical writer for a manufacturing facility.
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

USER_TEMPLATE = Template("""Write a task description for the following:

Task: $task_name
Department: $department
Equipment: $equipment
Relevant specifications: $specifications

Write ONLY the task description, no other commentary.""")


def generate_task_description(task_name, department, equipment, specifications):
    """Generate a standardized manufacturing task description."""
    user_prompt = USER_TEMPLATE.substitute(
        task_name=task_name,
        department=department,
        equipment=equipment,
        specifications=specifications,
    )

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1, "repeat_penalty": 1.2},
    )
    return response["message"]["content"]


# --- Test it with real manufacturing scenarios ---

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
        "specifications": "Drawing #SH-4402-Rev.B, tolerance +/-0.005 in",
    },
    {
        "task_name": "Perform daily forklift inspection",
        "department": "Warehouse",
        "equipment": "Toyota 8FGU25 forklift",
        "specifications": "OSHA 1910.178, Company SOP-FL-001",
    },
]

for task in test_tasks:
    print(f"\n{'=' * 60}")
    print(f"Input: {task['task_name']}")
    print(f"{'=' * 60}")
    description = generate_task_description(**task)
    print(description)

print(f"\n{'=' * 60}")
print("WHAT YOU BUILT")
print(f"{'=' * 60}")
print("A reusable, version-controlled prompt template that produces")
print("consistent task descriptions across different departments.")
print("\nNext steps in your journey:")
print("  Module 04 -> Get this output as structured JSON, not just text")
print("  Module 06 -> Feed it real reference documents via RAG")
print("  Module 09 -> Measure quality automatically instead of eyeballing it")
```

Run it:

```bash
python 03-prompt-engineering/step09_template.py
```

Look at all three outputs. They should follow the same format: CAPS title, numbered steps, action verbs, tool references, safety first for machinery tasks, documentation at the end. Three different tasks, one consistent style.

### Try these experiments:

**Change the department and watch the output adapt:**
Try changing "Warehouse" to "Clean Room" for the forklift task. What changes in the output?

**Add a few-shot example to the system prompt:**
Paste one of the generated descriptions back INTO the system prompt as an example. Does the consistency improve?

**Try a different model:**
Change `model="llama3.1:8b"` to `model="phi3:mini"`. How does the smaller model handle the same structured prompt? What breaks?

**Change one rule at a time:**
Remove the "8th-grade reading level" constraint. Run it. Does the language get more complex? Add it back. This is how you learn what each constraint actually does.

---

## Part 7: The Manufacturing Prompt Engineering Checklist

After working through all of that, here's what you now know:

### Techniques and When to Use Them

| Technique | What It Is | When to Use It |
|-----------|-----------|----------------|
| **Zero-shot** | Just ask for what you want | Quick tests, simple tasks |
| **System prompt** | Tell the model who it is and what rules to follow | Always -- this is your baseline |
| **Few-shot** | Show 2-3 examples of the output you want | When format and style consistency matter (your main use case) |
| **Chain-of-thought** | Ask the model to reason before answering | Complex tasks with multiple considerations |
| **Constrained output** | Specify exact format rules | When you need predictable structure |
| **Combined** | System prompt + few-shot + constraints | Production use -- this is your prompt template |

### The Iteration Process

1. Start with zero-shot. See what you get.
2. Add a system prompt with role and constraints. See what improves.
3. Add few-shot examples. See how style matching kicks in.
4. Tune one thing at a time. Save each version.
5. Compare versions. Pick the best one. That's your v1 for production.

This isn't a one-time thing. As you collect real task descriptions from your facility, those become better few-shot examples. As you learn what the model gets wrong, you add constraints to prevent it. The prompt evolves.

---

## Takeaways

1. **Few-shot prompting is your most powerful tool** -- showing the model examples of what you want beats describing it every time
2. **System prompts are non-negotiable** -- always define the role, format, style, and constraints
3. **Change one thing at a time** -- otherwise you don't know what helped and what hurt
4. **Save every version** -- prompt engineering is an iterative process, and you need the receipts
5. **Your prompt template is production code** -- version control it, test it, review it like any other code

## What's Next

Your prompts now generate good text -- but it comes back as a blob of unstructured strings. For a real system, you need the LLM to output **structured data** -- JSON with specific fields, predictable schemas, exactly the data your application can parse and store. Module 04 teaches you to constrain LLM output so it's machine-readable, not just human-readable.
