# Module 13: Evaluation Datasets & Benchmarks

## Goal
Build evaluation test infrastructure: a golden dataset, synthetic expansion, and a regression benchmark you can run any time to prove your system is getting better (or catch it getting worse).

---

## Why This Matters

Your evaluation is only as good as your test data. Let's build it.

Think of it this way. You would never qualify a weld process by testing one joint. You run coupon tests -- dozens of them, across different positions, thicknesses, and operators. Then you have a baseline. Every change gets tested against that baseline.

We are going to build the same thing for your LLM pipeline. A set of known-good examples, a way to expand them, and a benchmark that gives you a number. Before: 72%. After: 86%. That is the data your boss wants to see.

---

## Exercise 1: Building a Golden Dataset

A golden dataset is your collection of PERFECT examples. These are written or reviewed by domain experts -- the people who actually know what a good task description looks like.

Let's build one entry at a time.

### Your first golden example

```python
# 13-evaluation-datasets-and-benchmarks/ex1_golden_dataset.py
"""Build a golden dataset entry by entry."""

import json
```

Start with one example. Think about a task you know well:

```python
entry_1 = {
    "id": "gold-001",
    "task_name": "Inspect welded joints on Frame Assembly A",
    "department": "Quality Control",
    "context": "AWS D1.1, Form QC-107, fillet gauge required",
    "expected_output": """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete on the welding station.
2. Don required PPE: safety glasses, leather gloves, inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6 -- check for cracks, porosity, and undercut.
4. Measure weld size with fillet gauge -- minimum 6mm leg per drawing.
5. Record findings on Form QC-107. Tag defective joints with red HOLD tag and notify supervisor.""",
    "required_elements": ["numbered_steps", "ppe", "specification_reference", "form_reference", "action_verbs"],
    "difficulty": "medium",
}
```

Look at what makes this a good golden example:

- **The expected output is what a real expert would write.** Not LLM-generated, not a guess.
- **It has required_elements** -- a checklist of things the output MUST include.
- **It has difficulty** -- so you can see if your system handles hard cases vs easy ones.
- **It has context** -- the reference documents that should inform the answer.

Let's add a second one. Notice how different it is:

```python
entry_2 = {
    "id": "gold-002",
    "task_name": "Perform daily forklift pre-operation inspection",
    "department": "Warehouse",
    "context": "OSHA 1910.178, Company SOP-FL-001",
    "expected_output": """DAILY FORKLIFT PRE-OPERATION INSPECTION

1. Check tire condition and inflation pressure visually.
2. Test horn, headlights, backup alarm, and strobe light.
3. Verify hydraulic fluid level -- add if below MIN mark on dipstick.
4. Inspect mast chains for wear, damage, or excessive slack.
5. Test service brake and parking brake before loading.
6. Record inspection results on daily checklist per SOP-FL-001. Do not operate if any item fails -- report to maintenance.""",
    "required_elements": ["numbered_steps", "specification_reference", "action_verbs", "fail_criteria"],
    "difficulty": "easy",
}
```

This one is "easy" because forklift inspections are straightforward -- no complex measurements, no precision tolerances. Your system should nail this every time. If it cannot, something is wrong.

Let's add three more to round out the set:

```python
entry_3 = {
    "id": "gold-003",
    "task_name": "Set up CNC lathe for precision shaft machining",
    "department": "Machining",
    "context": "Drawing SH-4402-Rev.B, tolerance +/-0.005\", Haas ST-20",
    "expected_output": """SET UP CNC LATHE FOR PRECISION SHAFT MACHINING

1. Review drawing SH-4402-Rev.B and verify material is staged (confirm heat lot).
2. Load program from DNC server -- verify program number matches setup sheet.
3. Install 3-jaw chuck with soft jaws bored to part diameter +0.002".
4. Set tool offsets using the Renishaw tool setter. Verify against setup sheet (+/-0.001").
5. Run first article at 50% rapid, 75% feed override. Measure OD, length, and runout.
6. Record first article measurements on FAIR form. Proceed to production after QC approval.""",
    "required_elements": ["numbered_steps", "drawing_reference", "measurements", "action_verbs", "first_article"],
    "difficulty": "hard",
}

entry_4 = {
    "id": "gold-004",
    "task_name": "Replace hydraulic cylinder seals",
    "department": "Maintenance",
    "context": "Machine: 200-ton press, Seal kit P/N HK-200-SEAL, SOP-SAFE-001 for LOTO",
    "expected_output": """REPLACE HYDRAULIC CYLINDER SEALS

1. Perform lockout/tagout per SOP-SAFE-001. Bleed residual hydraulic pressure.
2. Disconnect hydraulic lines and cap all open ports to prevent contamination.
3. Remove cylinder from machine using overhead crane (rated >2 ton). Wear hard hat.
4. Disassemble cylinder, remove old seals. Inspect bore and rod for scoring (max 0.002" depth).
5. Install new seals from kit HK-200-SEAL. Lubricate with clean hydraulic fluid before installation.
6. Reassemble, reinstall, reconnect lines. Bleed air from circuit.
7. Remove LOTO, pressurize slowly. Check for leaks at zero, 50%, and 100% pressure. Log on PM-105.""",
    "required_elements": ["numbered_steps", "loto", "part_numbers", "measurements", "pressure_test"],
    "difficulty": "hard",
}

entry_5 = {
    "id": "gold-005",
    "task_name": "Calibrate digital caliper",
    "department": "Metrology",
    "context": "NIST-traceable gauge blocks, Calibration SOP-CAL-003, Form CAL-201",
    "expected_output": """CALIBRATE DIGITAL CALIPER

1. Clean caliper jaws and gauge blocks with lint-free cloth and isopropyl alcohol.
2. Zero the caliper with jaws fully closed -- verify display reads 0.000".
3. Measure gauge blocks at 0.500", 1.000", 2.000", and 4.000" (NIST-traceable set).
4. Record all readings on Form CAL-201. Tolerance: +/-0.001" at each point.
5. If any reading is out of tolerance, adjust per manufacturer instructions and re-test.
6. Apply calibration sticker with date, technician ID, and next-due date. Return to service.""",
    "required_elements": ["numbered_steps", "form_reference", "measurements", "tolerance", "calibration_sticker"],
    "difficulty": "medium",
}
```

Now save the whole set:

```python
golden_dataset = [entry_1, entry_2, entry_3, entry_4, entry_5]

output_path = "13-evaluation-datasets-and-benchmarks/golden_dataset.json"
with open(output_path, "w") as f:
    json.dump(golden_dataset, f, indent=2)

print(f"Golden dataset saved: {output_path}")
print(f"  {len(golden_dataset)} examples")
print(f"  Difficulties: {[e['difficulty'] for e in golden_dataset]}")
print(f"  Departments: {[e['department'] for e in golden_dataset]}")
```

Run it:

```bash
python 13-evaluation-datasets-and-benchmarks/ex1_golden_dataset.py
```

Check the JSON file that was created. Open it, look through it. This is your ground truth. Version control this file -- treat changes to it like code changes. A PR to update the golden dataset should be reviewed by a domain expert.

**Five examples is a good start, but it is not enough to be confident.** You need broader coverage. Let's generate more.

---

## Exercise 2: Generating Synthetic Test Cases

Writing golden examples by hand is slow. Five took us a while. Getting to 50 would take days. So let's use the LLM to generate synthetic test cases, then review them by hand.

The key word there is "review." Synthetic data is a starting point, not a finished product.

```python
# 13-evaluation-datasets-and-benchmarks/ex2_synthetic_dataset.py
"""Generate synthetic test cases using the LLM, then review them."""

from openai import OpenAI
import json

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
```

First, load the golden dataset as reference:

```python
with open("13-evaluation-datasets-and-benchmarks/golden_dataset.json") as f:
    golden = json.load(f)
```

Now let's ask the LLM to generate new tasks. We show it our golden examples so it understands the format:

```python
examples_text = "\n\n".join(
    f"Task: {g['task_name']}\nDept: {g['department']}\nContext: {g['context']}\nDifficulty: {g['difficulty']}"
    for g in golden[:3]
)

generation_prompt = f"""Based on these manufacturing task examples, generate 5 NEW and DIFFERENT tasks.

EXAMPLES:
{examples_text}

Generate 5 new tasks covering these departments: assembly, painting, shipping, electrical, quality lab.

Return a JSON array. Each item needs:
- task_name: descriptive name
- department: which department
- context: relevant specs, forms, standards
- difficulty: "easy", "medium", or "hard"
- required_elements: list of what a good description should include

Return ONLY the JSON array, no other text."""
```

Let's run it:

```python
print("Generating synthetic test cases...\n")
response = llm.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": generation_prompt}],
    response_format={"type": "json_object"},
    temperature=0.7,
)

try:
    synthetic = json.loads(response.choices[0].message.content)

    # Handle if the LLM wraps it in a key
    if isinstance(synthetic, dict):
        for key in synthetic:
            if isinstance(synthetic[key], list):
                synthetic = synthetic[key]
                break

    print(f"Generated {len(synthetic)} synthetic test cases:\n")
    for task in synthetic:
        print(f"  [{task.get('difficulty', '?'):6s}] {task.get('task_name', '?')}")
        print(f"          Dept: {task.get('department', '?')}")
        print(f"          Context: {task.get('context', '?')[:60]}...")
        print()

except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print(f"Raw output: {response.choices[0].message.content[:200]}")
    synthetic = []
```

**Stop and look at these.** Before you save them, ask yourself:

- Do the tasks make sense for a real factory?
- Are the contexts realistic? (Real spec numbers, real form names?)
- Is the difficulty rating accurate?
- Are there any duplicates of your golden examples?

This is the human review step. In production, you would have a domain expert spend 10 minutes checking these. Let's save the ones that pass review:

```python
if synthetic:
    output_path = "13-evaluation-datasets-and-benchmarks/synthetic_dataset.json"
    with open(output_path, "w") as f:
        json.dump(synthetic, f, indent=2)
    print(f"Saved to {output_path}")
    print("IMPORTANT: Review these before using them in benchmarks!")
    print("Remove any that look unrealistic or duplicate your golden set.")
```

Now let's generate expected outputs for the synthetic tasks too. This gives us something to compare against:

```python
print("\n--- Generating expected outputs for synthetic tasks ---\n")
for i, task in enumerate(synthetic[:3]):  # Do 3 to save time
    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {"role": "system", "content": "You are a senior manufacturing technical writer. Write 3-5 numbered steps with safety notes and spec references."},
            {"role": "user", "content": f"Task: {task['task_name']}\nContext: {task.get('context', 'N/A')}"},
        ],
        temperature=0.0,
    )
    task["expected_output"] = response.choices[0].message.content
    print(f"Generated output for: {task['task_name']}")
    print(f"  Preview: {task['expected_output'][:80]}...")
    print()

# Re-save with expected outputs
if synthetic:
    with open("13-evaluation-datasets-and-benchmarks/synthetic_dataset.json", "w") as f:
        json.dump(synthetic, f, indent=2)
    print("Updated synthetic dataset with expected outputs.")
```

**The workflow is: generate, review, refine, save.** Never trust synthetic data blindly. The LLM is good at creating plausible-looking tasks, but a domain expert catches things like "that is not how you calibrate that instrument" or "that spec number does not exist."

---

## Exercise 3: Your First Regression Benchmark

Now you have test data. Let's use it. A regression benchmark does one thing: run your system against a fixed set of inputs and give you a score. Then when you change something -- a prompt, a model, a retrieval parameter -- you re-run and compare.

This is the before/after comparison that proves your changes are improvements.

```python
# 13-evaluation-datasets-and-benchmarks/ex3_regression_benchmark.py
"""Run a regression benchmark -- get a score, change something, get a new score."""

from openai import OpenAI
import json
import re
from datetime import datetime

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
```

First, let's build the evaluation function. This checks the things we care about:

```python
def evaluate_output(generated: str, task: dict) -> dict:
    """Score a generated output against our quality criteria."""
    scores = {}

    # Does it have numbered steps?
    steps = re.findall(r'^\s*\d+[\.\)]', generated, re.MULTILINE)
    scores["has_steps"] = len(steps) >= 3

    # Is the length reasonable?
    word_count = len(generated.split())
    scores["good_length"] = 30 <= word_count <= 200

    # Check for required elements
    element_checks = {
        "numbered_steps": len(steps) >= 3,
        "ppe": any(w in generated.lower() for w in ["ppe", "glasses", "gloves", "helmet", "boots"]),
        "specification_reference": bool(re.search(r'[A-Z]{2,}[\s-]\d', generated)),
        "form_reference": bool(re.search(r'[Ff]orm\s+[A-Z]', generated)),
        "action_verbs": any(w in generated.lower() for w in ["inspect", "verify", "check", "install", "remove", "record", "measure"]),
        "loto": any(w in generated.lower() for w in ["lockout", "tagout", "loto"]),
        "measurements": bool(re.search(r'\d+\.?\d*\s*(mm|Nm|"|inch|PSI)', generated)),
        "drawing_reference": bool(re.search(r'[Dd]rawing', generated)),
        "fail_criteria": any(w in generated.lower() for w in ["fail", "reject", "hold", "do not"]),
        "part_numbers": bool(re.search(r'[A-Z]{1,3}-\d{3,}', generated)),
        "tolerance": bool(re.search(r'[+±]', generated)),
        "first_article": "first article" in generated.lower(),
        "calibration_sticker": "calibration" in generated.lower(),
        "pressure_test": "pressure" in generated.lower(),
    }

    required = task.get("required_elements", [])
    if required:
        passed = sum(1 for r in required if element_checks.get(r, False))
        scores["required_elements"] = passed / len(required)
    else:
        scores["required_elements"] = 0.5  # No requirements specified

    # Overall: average of all scores
    all_values = [float(v) for v in scores.values()]
    scores["overall"] = sum(all_values) / len(all_values)

    return scores
```

Now the benchmark runner. It takes a system prompt and runs every golden example through it:

```python
def run_benchmark(golden_path: str, system_prompt: str, model: str = "gemma3:12b") -> dict:
    """Run the benchmark with a given system prompt."""
    with open(golden_path) as f:
        golden = json.load(f)

    results = []
    for task in golden:
        response = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task['task_name']}\nContext: {task['context']}"},
            ],
            temperature=0.0,
        )
        generated = response.choices[0].message.content
        scores = evaluate_output(generated, task)
        scores["task_id"] = task["id"]
        scores["task_name"] = task["task_name"]
        results.append(scores)

    # Summary
    overalls = [r["overall"] for r in results]
    return {
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt[:80] + "...",
        "model": model,
        "avg_score": sum(overalls) / len(overalls),
        "min_score": min(overalls),
        "max_score": max(overalls),
        "passing": sum(1 for o in overalls if o >= 0.7),
        "total": len(overalls),
        "per_task": results,
    }
```

Here is where it gets interesting. Let's run two benchmarks -- one with a bad prompt, one with a good prompt -- and compare:

```python
golden_path = "13-evaluation-datasets-and-benchmarks/golden_dataset.json"

# Run 1: Minimal prompt (the lazy approach)
print("=== Benchmark Run 1: Minimal Prompt ===")
print("Prompt: 'Write a manufacturing task description.'\n")

run1 = run_benchmark(
    golden_path,
    system_prompt="Write a manufacturing task description.",
)

print(f"  Average score: {run1['avg_score']:.0%}")
print(f"  Passing (>=70%): {run1['passing']}/{run1['total']}")
for r in run1["per_task"]:
    status = "PASS" if r["overall"] >= 0.7 else "FAIL"
    print(f"    [{status}] {r['task_name'][:45]:45s} {r['overall']:.0%}")
```

Now the good prompt:

```python
# Run 2: Detailed prompt (the engineered approach)
print("\n=== Benchmark Run 2: Detailed Prompt ===")
detailed_prompt = """You are a senior manufacturing technical writer at an ISO 9001 facility.
Write task descriptions with:
- Numbered steps (3-7 steps), each starting with an action verb
- PPE and safety requirements where applicable
- References to specific forms, specifications, and part numbers
- Measurable acceptance criteria
- Active voice, 8th-grade reading level, 50-120 words"""

print(f"Prompt: '{detailed_prompt[:60]}...'\n")

run2 = run_benchmark(golden_path, system_prompt=detailed_prompt)

print(f"  Average score: {run2['avg_score']:.0%}")
print(f"  Passing (>=70%): {run2['passing']}/{run2['total']}")
for r in run2["per_task"]:
    status = "PASS" if r["overall"] >= 0.7 else "FAIL"
    print(f"    [{status}] {r['task_name'][:45]:45s} {r['overall']:.0%}")
```

Now the comparison -- this is the payoff:

```python
# Compare
delta = run2["avg_score"] - run1["avg_score"]
print("\n=== Comparison ===")
print(f"  Run 1 (minimal):  {run1['avg_score']:.0%}")
print(f"  Run 2 (detailed): {run2['avg_score']:.0%}")
print(f"  Delta:            {delta:+.0%}")

if delta > 0.05:
    print(f"  Verdict: IMPROVEMENT. The detailed prompt is better by {delta:.0%}.")
elif delta < -0.05:
    print(f"  Verdict: REGRESSION. The detailed prompt is worse by {abs(delta):.0%}!")
else:
    print(f"  Verdict: NO SIGNIFICANT CHANGE. Delta is within 5%.")
```

**THIS is how you prove improvements.** Not "I think it looks better" but "Run 1: 72%. Run 2: 86%. The detailed prompt improved scores by 14 percentage points." That is the kind of evidence that justifies spending time on prompt engineering.

Save the results so you can track over time:

```python
# Save results for historical tracking
results_path = "13-evaluation-datasets-and-benchmarks/benchmark_results.json"
history = {"runs": [run1, run2]}
with open(results_path, "w") as f:
    json.dump(history, f, indent=2, default=str)
print(f"\nResults saved to {results_path}")
print("Run this again after any change to track improvements over time.")
```

---

## Exercise 4: Making the Benchmark Easy to Run

A benchmark you do not run is useless. Let's make it a one-liner.

```python
# 13-evaluation-datasets-and-benchmarks/ex4_benchmark_runner.py
"""A reusable benchmark runner -- one command to get your score."""

from openai import OpenAI
import json
import re
import sys
from datetime import datetime

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def evaluate_output(generated: str, task: dict) -> dict:
    """Score a generated output against quality criteria."""
    steps = re.findall(r'^\s*\d+[\.\)]', generated, re.MULTILINE)
    element_checks = {
        "numbered_steps": len(steps) >= 3,
        "ppe": any(w in generated.lower() for w in ["ppe", "safety", "glasses", "gloves", "helmet"]),
        "specification_reference": bool(re.search(r'[A-Z]{2,}[\s-]\d', generated)),
        "form_reference": bool(re.search(r'[Ff]orm\s+[A-Z]', generated)),
        "action_verbs": any(w in generated.lower() for w in ["inspect", "verify", "check", "install", "record", "measure"]),
    }
    required = task.get("required_elements", [])
    if required:
        required_score = sum(1 for r in required if element_checks.get(r, False)) / len(required)
    else:
        required_score = 0.5

    has_steps = len(steps) >= 3
    good_length = 30 <= len(generated.split()) <= 200
    overall = (float(has_steps) + float(good_length) + required_score) / 3
    return {"has_steps": has_steps, "good_length": good_length, "required_elements": required_score, "overall": overall}


def run(prompt_file: str = None):
    """Run the benchmark."""
    golden_path = "13-evaluation-datasets-and-benchmarks/golden_dataset.json"
    with open(golden_path) as f:
        golden = json.load(f)

    # Load system prompt from file or use default
    if prompt_file:
        with open(prompt_file) as f:
            system_prompt = f.read().strip()
        print(f"Using prompt from: {prompt_file}")
    else:
        system_prompt = "You are a manufacturing technical writer. Write numbered steps with safety and spec references."
        print("Using default prompt.")

    print(f"Running {len(golden)} test cases...\n")

    total_score = 0
    passing = 0
    for task in golden:
        response = llm.chat.completions.create(
            model="gemma3:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task['task_name']}\nContext: {task['context']}"},
            ],
            temperature=0.0,
        )
        scores = evaluate_output(response.choices[0].message.content, task)
        total_score += scores["overall"]
        if scores["overall"] >= 0.7:
            passing += 1
        status = "PASS" if scores["overall"] >= 0.7 else "FAIL"
        print(f"  [{status}] {task['task_name'][:50]:50s} {scores['overall']:.0%}")

    avg = total_score / len(golden)
    print(f"\n  SCORE: {avg:.0%} ({passing}/{len(golden)} passing)")
    return avg


if __name__ == "__main__":
    prompt_file = sys.argv[1] if len(sys.argv) > 1 else None
    run(prompt_file)
```

Now you can run your benchmark like this:

```bash
# With default prompt
python 13-evaluation-datasets-and-benchmarks/ex4_benchmark_runner.py

# With a custom prompt from a file
echo "You are a manufacturing writer. Write 3-5 steps." > my_prompt.txt
python 13-evaluation-datasets-and-benchmarks/ex4_benchmark_runner.py my_prompt.txt
```

One command. One number. That is what makes a benchmark actually useful.

---

## Takeaways

1. **Golden datasets are your ground truth** -- expert-written, version-controlled, reviewed like code
2. **Start with 5 examples, grow as you find edge cases** -- you do not need 500 on day one
3. **Synthetic data scales coverage** but always needs human review -- the LLM generates plausible fakes
4. **Regression benchmarks give you a number** -- before and after, no more guessing
5. **Make benchmarks easy to run** -- if it takes more than one command, people will not run it
6. **Version your eval data in git** -- it is as important as your code

## What's Next

Phase 3 is complete -- you have a full evaluation infrastructure. Module 14 starts Phase 4 (advanced techniques) with fine-tuning: when RAG plus prompting is not enough, and you need to teach the model new behaviors.
