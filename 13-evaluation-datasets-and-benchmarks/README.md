# Module 13: Evaluation Datasets & Benchmarks

## Goal
Build evaluation test infrastructure: a golden dataset, synthetic expansion, and a regression benchmark you can run any time to prove your system is getting better (or catch it getting worse).

---

## Why This Matters

Your evaluation is only as good as your test data. Let's build it.

Think of it this way. You would never evaluate a draft class by watching one game of film. You grade dozens of prospects, across different positions, schemes, and competition levels. Then you have a baseline. Every new piece of film gets evaluated against that baseline.

We are going to build the same thing for your LLM pipeline. A set of known-good examples, a way to expand them, and a benchmark that gives you a number. Before: 72%. After: 86%. That is the data your front office wants to see.

---

## Exercise 1: Building a Golden Dataset

A golden dataset is your collection of PERFECT examples. These are written or reviewed by domain experts -- the people who actually know what a good scouting report looks like.

Let's build one entry at a time.

### Your first golden example

```python
# 13-evaluation-datasets-and-benchmarks/ex1_golden_dataset.py
"""Build a golden dataset entry by entry."""

import json
```

Start with one example. Think about a prospect you know well:

```python
entry_1 = {
    "id": "gold-001",
    "task_name": "Evaluate QB pocket passing ability",
    "department": "Quarterback Scouting",
    "context": "QB-101 scouting report, 68% completion, 62 mph arm, 2.3s release",
    "expected_output": """QB POCKET PASSING EVALUATION

1. Review game film focusing on intermediate routes (15-25 yards) where prospect excels.
2. Chart completion percentage by route type -- 68% overall with elite accuracy on crossing routes.
3. Time release from snap to throw -- 2.3-second average indicates quick processing.
4. Evaluate pre-snap reads -- prospect consistently identifies defensive coverages before the snap.
5. Note weakness: locks onto first read under pressure. Track sack rate and pressure-to-scramble ratio.""",
    "required_elements": ["numbered_steps", "measurables", "film_reference", "weakness_note", "action_verbs"],
    "difficulty": "medium",
}
```

Look at what makes this a good golden example:

- **The expected output is what a real scout would write.** Not LLM-generated, not a guess.
- **It has required_elements** -- a checklist of things the output MUST include.
- **It has difficulty** -- so you can see if your system handles hard cases vs easy ones.
- **It has context** -- the scouting documents that should inform the answer.

Let's add a second one. Notice how different it is:

```python
entry_2 = {
    "id": "gold-002",
    "task_name": "Grade running back explosiveness and receiving",
    "department": "Running Back Scouting",
    "context": "RB-201 scouting report, 4.38 40-yard dash, 3.8 YAC, 45 receptions",
    "expected_output": """RUNNING BACK EXPLOSIVENESS AND RECEIVING GRADE

1. Review combine data -- 4.38 40-yard dash confirms top-tier straight-line speed.
2. Chart yards after contact -- 3.8 average shows ability to break tackles at the second level.
3. Evaluate vision on film -- prospect finds cutback lanes consistently, reads blocks well.
4. Grade receiving ability -- 45 receptions out of backfield shows viable third-down role.
5. Flag weakness: pass protection and blitz pickup are below average. Will need coaching at next level.""",
    "required_elements": ["numbered_steps", "measurables", "action_verbs", "weakness_note"],
    "difficulty": "easy",
}
```

This one is "easy" because running back evaluations are straightforward -- clear measurables, defined role. Your system should nail this every time. If it cannot, something is wrong.

Let's add three more to round out the set:

```python
entry_3 = {
    "id": "gold-003",
    "task_name": "Evaluate wide receiver route running and separation",
    "department": "Wide Receiver Scouting",
    "context": "WR-301 scouting report, 4.42 speed, 38-inch vertical, 2.1% drop rate",
    "expected_output": """WIDE RECEIVER ROUTE RUNNING AND SEPARATION EVALUATION

1. Break down route tree on film -- prospect runs full tree from both slot and outside alignments.
2. Grade separation ability -- elite at creating space with crisp breaks and deception.
3. Verify combine measurables -- 4.42 speed and 38-inch vertical confirm high-end athleticism.
4. Chart hands reliability -- 2.1% drop rate is elite among this draft class.
5. Note weakness: struggles with press coverage at the line of scrimmage. Needs to add strength to counter physical corners.""",
    "required_elements": ["numbered_steps", "film_reference", "measurables", "action_verbs", "weakness_note"],
    "difficulty": "hard",
}

entry_4 = {
    "id": "gold-004",
    "task_name": "Grade offensive lineman pass protection",
    "department": "Offensive Line Scouting",
    "context": "OL-401 scouting report, 34-inch arms, 82.5 run blocking grade, 2 sacks in 580 snaps",
    "expected_output": """OFFENSIVE LINEMAN PASS PROTECTION GRADE

1. Review pass protection snaps on film -- 2 sacks allowed in 580 snaps indicates elite anchor.
2. Measure arm length -- 34-inch arms provide excellent punch range and leverage advantage.
3. Evaluate lateral movement -- quick feet allow prospect to mirror speed rushers effectively.
4. Grade run blocking -- 82.5/100 shows above-average ability to create movement at the point of attack.
5. Note weakness: combo blocks are inconsistent. Struggles to transition from double team to second-level linebacker.
6. Project role: Day 1 starter at guard or right tackle. Pass protection is NFL-ready.""",
    "required_elements": ["numbered_steps", "measurables", "film_reference", "weakness_note", "projection"],
    "difficulty": "hard",
}

entry_5 = {
    "id": "gold-005",
    "task_name": "Analyze defensive scheme tendencies",
    "department": "Defensive Scouting",
    "context": "DEF-501 scouting report, Cover-3 base, press corners, pattern-match zone on 3rd down",
    "expected_output": """DEFENSIVE SCHEME TENDENCY ANALYSIS

1. Chart base defense -- Cover-3 with single-high safety is the primary look on early downs.
2. Evaluate corner technique -- press coverage at the line with trail technique underneath.
3. Break down third-down package -- pattern-match zone replaces pure man coverage, aggressive nickel blitz.
4. Identify tendency: team runs aggressive nickel blitz on 3rd-and-medium (4-7 yards).
5. Note weakness: crossing routes consistently beat the zone coverage. Quick slants and mesh concepts are the primary exploits.""",
    "required_elements": ["numbered_steps", "scheme_reference", "tendencies", "action_verbs", "weakness_note"],
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
print(f"  Positions: {[e['department'] for e in golden_dataset]}")
```

Run it:

```bash
python 13-evaluation-datasets-and-benchmarks/ex1_golden_dataset.py
```

Check the JSON file that was created. Open it, look through it. This is your ground truth. Version control this file -- treat changes to it like code changes. A PR to update the golden dataset should be reviewed by a scout or analyst who knows football.

**Five examples is a good start, but it is not enough to be confident.** You need broader coverage. Let's generate more.

---

## Exercise 2: Generating Synthetic Test Cases

Writing golden examples by hand is slow. Five took us a while. Getting to 50 would take days. So let's use the LLM to generate synthetic scouting Q&A pairs, then review them by hand.

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

Now let's ask the LLM to generate new scouting evaluations. We show it our golden examples so it understands the format:

```python
examples_text = "\n\n".join(
    f"Task: {g['task_name']}\nPosition: {g['department']}\nContext: {g['context']}\nDifficulty: {g['difficulty']}"
    for g in golden[:3]
)

generation_prompt = f"""Based on these football scouting evaluation examples, generate 5 NEW and DIFFERENT evaluations.

EXAMPLES:
{examples_text}

Generate 5 new evaluations covering these positions: tight end, safety, edge rusher, interior DL, cornerback.

Return a JSON array. Each item needs:
- task_name: descriptive evaluation name
- department: which position group
- context: relevant scouting data, measurables, film notes
- difficulty: "easy", "medium", or "hard"
- required_elements: list of what a good scouting report should include

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

- Do the evaluations make sense for real NFL scouting?
- Are the contexts realistic? (Real measurables, real film observations?)
- Is the difficulty rating accurate?
- Are there any duplicates of your golden examples?

This is the human review step. In production, you would have a scout spend 10 minutes checking these. Let's save the ones that pass review:

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
print("\n--- Generating expected outputs for synthetic evaluations ---\n")
for i, task in enumerate(synthetic[:3]):  # Do 3 to save time
    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {"role": "system", "content": "You are a senior NFL scout. Write 3-5 numbered evaluation points with measurables, film notes, and prospect grades."},
            {"role": "user", "content": f"Evaluation: {task['task_name']}\nContext: {task.get('context', 'N/A')}"},
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

**The workflow is: generate, review, refine, save.** Never trust synthetic data blindly. The LLM is good at creating plausible-looking evaluations, but a real scout catches things like "that 40 time is unrealistic for an interior DL" or "that coverage scheme does not exist."

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
        "measurables": any(w in generated.lower() for w in ["40-yard", "vertical", "arm strength", "speed", "release", "mph", "dash"]),
        "film_reference": any(w in generated.lower() for w in ["film", "tape", "game", "snap", "rep"]),
        "scheme_reference": any(w in generated.lower() for w in ["cover", "zone", "man", "blitz", "press", "nickel"]),
        "action_verbs": any(w in generated.lower() for w in ["evaluate", "grade", "chart", "review", "break down", "measure", "project"]),
        "weakness_note": any(w in generated.lower() for w in ["weakness", "concern", "struggles", "below average", "needs"]),
        "tendencies": any(w in generated.lower() for w in ["tendency", "pattern", "consistently", "frequently", "primary"]),
        "projection": any(w in generated.lower() for w in ["project", "starter", "role", "day 1", "round", "ceiling"]),
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
print("Prompt: 'Write an NFL scouting report.'\n")

run1 = run_benchmark(
    golden_path,
    system_prompt="Write an NFL scouting report.",
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
detailed_prompt = """You are a senior NFL draft analyst preparing scouting evaluations.
Write scouting reports with:
- Numbered evaluation points (3-7 points), each starting with an action verb
- Combine measurables and game stats where available
- References to specific film, games, and scouting data
- Identified weaknesses and projection to the next level
- Active voice, clear and direct, 50-120 words"""

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
        "measurables": any(w in generated.lower() for w in ["40-yard", "vertical", "arm strength", "speed", "release", "mph"]),
        "film_reference": any(w in generated.lower() for w in ["film", "tape", "game", "snap", "rep"]),
        "scheme_reference": any(w in generated.lower() for w in ["cover", "zone", "man", "blitz", "press"]),
        "action_verbs": any(w in generated.lower() for w in ["evaluate", "grade", "chart", "review", "break down", "measure"]),
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
        system_prompt = "You are an NFL draft analyst. Write numbered evaluation points with measurables and film references."
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
echo "You are an NFL scout. Write 3-5 evaluation points." > my_prompt.txt
python 13-evaluation-datasets-and-benchmarks/ex4_benchmark_runner.py my_prompt.txt
```

One command. One number. That is what makes a benchmark actually useful.

---

## Takeaways

1. **Golden datasets are your ground truth** -- expert-written, version-controlled, reviewed like code
2. **Start with 5 examples, grow as you find edge cases** -- you do not need 500 on day one
3. **Synthetic data scales coverage** but always needs scout review -- the LLM generates plausible fakes
4. **Regression benchmarks give you a number** -- before and after, no more guessing
5. **Make benchmarks easy to run** -- if it takes more than one command, people will not run it
6. **Version your eval data in git** -- it is as important as your code

## What's Next

Phase 3 is complete -- you have a full evaluation infrastructure. Module 14 starts Phase 4 (advanced techniques) with fine-tuning: when RAG plus prompting is not enough, and you need to teach the model new behaviors.
