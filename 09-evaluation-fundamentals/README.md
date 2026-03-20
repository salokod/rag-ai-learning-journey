# Module 09: Evaluation Fundamentals

## Goal
Answer the question every stakeholder will ask: **"How do we know the AI is doing a good job?"** By the end of this module, you will have a concrete, numbers-backed answer.

---

## The Scene

Your boss walks into your office.

> **Boss:** "So this AI thing you've been building for task descriptions... how do we know it's actually good?"
>
> **You (today):** "Uh, they look pretty good to me?"
>
> **You (after this module):** "We tested 50 task descriptions against our quality rubric. The AI scored 87% on format compliance, 92% on safety inclusion, and 78% on specificity. The main gap is referencing specific form numbers -- I'm adding more reference docs to close that gap."

Which person gets the project approved?

Let's build that answer. Start simple.

---

## Part 1: Rule-Based Checks (Free and Instant)

Before anything fancy, let's ask the simplest possible question: does the AI output even *look* right?

### Step 1: One tiny check

Open a Python shell or create a new file:

```python
# 09-evaluation-fundamentals/step1_one_check.py

good_task = """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete on the welding station.
2. Don required PPE: safety glasses, leather gloves, inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6.
4. Measure weld size with fillet gauge -- minimum 6mm leg per drawing SH-4402.
5. Record findings on Form QC-107. Tag defective joints with red HOLD tag."""

bad_task = "Check the welds."
```

Run this:

```python
import re

def has_numbered_steps(text):
    steps = re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)
    return len(steps)

print(has_numbered_steps(good_task))  # What number do you see?
print(has_numbered_steps(bad_task))   # And this one?
```

You should see `5` and `0`. That single function just told you something useful: one output has structure, the other doesn't. And it ran in microseconds. No LLM needed.

### Step 2: Check for safety mentions

Now let's add another check. In manufacturing, safety content is non-negotiable:

```python
def count_safety_mentions(text):
    safety_terms = [
        "ppe", "safety", "glasses", "gloves", "helmet",
        "lockout", "tagout", "caution", "warning", "hazard",
        "protective", "fire watch"
    ]
    text_lower = text.lower()
    return sum(1 for term in safety_terms if term in text_lower)

print(count_safety_mentions(good_task))  # Try it
print(count_safety_mentions(bad_task))   # And this
```

The good task should score 4+ (lockout, tagout, PPE, glasses, gloves). The bad task? Zero. You now have two checks that cost nothing and run instantly.

### Step 3: Action verbs

Manufacturing task descriptions should start steps with action verbs -- "inspect", "verify", "measure" -- not passive language like "should be checked":

```python
def count_action_verbs(text):
    verbs = [
        "inspect", "verify", "check", "install", "remove",
        "clean", "record", "document", "apply", "ensure",
        "confirm", "test", "measure", "adjust", "replace",
        "tighten", "calibrate", "don", "tag", "notify"
    ]
    text_lower = text.lower()
    return sum(1 for v in verbs if v in text_lower)

print(count_action_verbs(good_task))
print(count_action_verbs(bad_task))
```

Notice how each check is just a few lines. Nothing complicated. But together they start to paint a picture.

### Step 4: Specific references

Real task descriptions cite form numbers, specs, and measurements. Vague ones don't:

```python
def has_specific_references(text):
    has_form_ref = bool(re.search(r'[A-Z]{2,}-\d{2,}', text))
    has_measurement = bool(re.search(
        r'\d+\.?\d*\s*(mm|cm|inch|"|Nm|PSI|degrees|CFH|RPM)', text
    ))
    return has_form_ref, has_measurement

form, meas = has_specific_references(good_task)
print(f"Form numbers: {form}, Measurements: {meas}")

form, meas = has_specific_references(bad_task)
print(f"Form numbers: {form}, Measurements: {meas}")
```

The good task has both (QC-107, SH-4402, AWS D1.1, 6mm). The bad task has neither.

### Step 5: Combine them into a score

Now let's put it all together. Each check becomes a pass/fail, and we get a percentage:

```python
def heuristic_score(text):
    checks = {
        "has_numbered_steps": has_numbered_steps(text) >= 3,
        "appropriate_length": 30 <= len(text.split()) <= 200,
        "uses_action_verbs": count_action_verbs(text) >= 2,
        "mentions_safety": count_safety_mentions(text) >= 1,
        "has_form_numbers": has_specific_references(text)[0],
        "has_measurements": has_specific_references(text)[1],
    }
    score = sum(checks.values()) / len(checks)
    return checks, score

checks, score = heuristic_score(good_task)
print(f"Good task: {score:.0%}")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")

print()

checks, score = heuristic_score(bad_task)
print(f"Bad task: {score:.0%}")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")
```

You should see something like 100% vs 0%. That's a clear signal, and it took zero LLM calls.

### Step 6: Test the middle ground

What about a "mediocre" task description? Let's see if our scoring catches it:

```python
mediocre_task = """Weld Inspection

Look at the welds and make sure they are good. Check for any problems.
If something looks wrong, tell someone about it.
Make sure to write it down when you're done."""

checks, score = heuristic_score(mediocre_task)
print(f"Mediocre task: {score:.0%}")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")
```

Notice what happens: it passes on length and maybe action verbs (it has "check" and "inspect" synonyms), but fails on numbered steps, safety, and references. It correctly lands between good and bad.

**Key insight:** Rules catch structure but they can NOT catch quality. "Inspect the weld. Tighten the bolt. Record the data." would pass every check, but it's still a terrible task description. We need something smarter.

---

## Part 2: LLM-as-Judge

The idea: use an LLM to evaluate another LLM's output. The judge LLM reads the task description and scores it, just like a human reviewer would -- but at machine speed.

### Step 7: Score one thing -- clarity

Let's start with a single criterion. Keep it focused:

```python
# 09-evaluation-fundamentals/step7_llm_judge.py
import ollama
import json

def judge_clarity(text):
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate manufacturing task descriptions. "
                    "Score the CLARITY on a 0-10 scale. "
                    "0 = confusing, 10 = crystal clear for any operator. "
                    'Return JSON: {"clarity": <score>, "reason": "<why>"}'
                ),
            },
            {"role": "user", "content": text},
        ],
        format="json",
        options={"temperature": 0.0},
    )
    return json.loads(response["message"]["content"])
```

Try it on your good task:

```python
result = judge_clarity(good_task)
print(f"Clarity: {result['clarity']}/10")
print(f"Reason: {result['reason']}")
```

Now the bad task:

```python
result = judge_clarity(bad_task)
print(f"Clarity: {result['clarity']}/10")
print(f"Reason: {result['reason']}")
```

See the difference? The LLM can judge *meaning*, not just structure. It understands that "Check the welds" is vague in a way that regex never could.

### Step 8: Add more criteria

Now let's expand to a full rubric. Notice how we build on the same pattern:

```python
JUDGE_PROMPT = """You evaluate manufacturing task descriptions.
Score on each criterion (0-10):

1. CLARITY: Can any operator understand what to do?
2. COMPLETENESS: Are all necessary steps included?
3. SAFETY: Are relevant safety precautions mentioned?
4. SPECIFICITY: Does it cite tools, specs, form numbers?
5. PROFESSIONALISM: Does it read like a real manufacturing document?

Return JSON:
{
  "clarity": {"score": N, "reason": "..."},
  "completeness": {"score": N, "reason": "..."},
  "safety": {"score": N, "reason": "..."},
  "specificity": {"score": N, "reason": "..."},
  "professionalism": {"score": N, "reason": "..."},
  "overall": N
}"""

def judge_full(text):
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": text},
        ],
        format="json",
        options={"temperature": 0.0},
    )
    return json.loads(response["message"]["content"])
```

Run it:

```python
result = judge_full(good_task)
for criterion in ["clarity", "completeness", "safety", "specificity", "professionalism"]:
    entry = result.get(criterion, {})
    score = entry.get("score", "?")
    reason = entry.get("reason", "")
    bar = "#" * int(score) if isinstance(score, (int, float)) else ""
    print(f"  {criterion:15s}: {score}/10  {bar}")
    print(f"  {'':15s}  -> {reason}")
print(f"\n  Overall: {result.get('overall', '?')}/10")
```

Now run the same thing on the mediocre task:

```python
result = judge_full(mediocre_task)
for criterion in ["clarity", "completeness", "safety", "specificity", "professionalism"]:
    entry = result.get(criterion, {})
    score = entry.get("score", "?")
    reason = entry.get("reason", "")
    print(f"  {criterion:15s}: {score}/10 -> {reason}")
```

Compare the scores side by side. The LLM judge catches nuance that rules miss: it knows "tell someone about it" is unprofessional, and it knows the mediocre version omits critical safety steps.

### Step 9: Watch out for bias

One gotcha -- LLM judges tend to prefer longer, more verbose outputs. Let's see:

```python
verbose_but_bad = """COMPREHENSIVE WELDING INSPECTION PROCEDURE FOR QUALITY ASSURANCE

1. Prior to commencing any inspection activities, it is imperative that the
   designated inspection personnel verify and confirm that all applicable
   lockout/tagout procedures have been properly and thoroughly executed.
2. The inspection technician shall don all requisite personal protective
   equipment including but not limited to safety glasses, leather gloves,
   and inspection magnification devices.
3. A comprehensive visual examination shall be conducted."""

result = judge_full(verbose_but_bad)
print(f"Verbose-but-bad overall: {result.get('overall', '?')}/10")
```

If the score is surprisingly high, that's the verbosity bias at work. The text sounds professional but says very little of substance. Keep this in mind when interpreting LLM judge scores.

---

## Part 3: A/B Comparison

Sometimes you don't need absolute scores. You just need to know: **is version A better than version B?**

### Step 10: Head-to-head comparison

```python
# 09-evaluation-fundamentals/step10_ab_compare.py
import ollama
import json

COMPARE_PROMPT = """You are comparing two manufacturing task descriptions.
Which is BETTER for actual use on a manufacturing floor?

Consider:
- Can an operator actually follow it?
- Are safety requirements covered?
- Are specific references included?

TASK: {task}

VERSION A:
{version_a}

VERSION B:
{version_b}

Return JSON:
{{
  "winner": "A" or "B",
  "confidence": "high" / "medium" / "low",
  "key_difference": "the main reason one is better"
}}"""

def compare_versions(task, version_a, version_b):
    prompt = COMPARE_PROMPT.format(
        task=task, version_a=version_a, version_b=version_b
    )
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0},
    )
    return json.loads(response["message"]["content"])
```

Let's test it -- RAG-enhanced vs. plain LLM output:

```python
task = "Replace worn conveyor belt rollers"

version_a = """Replace Conveyor Belt Rollers

1. Lock out conveyor system per LOTO procedure SOP-SAFE-001.
2. Don PPE: safety glasses, leather gloves, steel-toe boots.
3. Remove belt tension using take-up adjustment (manual Section 3.2).
4. Remove worn rollers, inspect bearing seats for damage.
5. Install new rollers (P/N CR-200 series), verify free rotation.
6. Restore belt tension to 2-3% elongation per specification.
7. Run conveyor empty for 5 minutes, check tracking and alignment.
8. Document replacement on Form PM-105."""

version_b = """Change the old rollers on the conveyor belt.

Turn off the conveyor first. Take out the bad rollers and put in
new ones. Make sure the belt works when you turn it back on.
Write down what you did."""

result = compare_versions(task, version_a, version_b)
print(f"Winner: Version {result['winner']}")
print(f"Confidence: {result['confidence']}")
print(f"Key difference: {result['key_difference']}")
```

No surprise which wins. But notice the power here: you can use this to compare prompt versions, before/after RAG improvements, or different models on the same task. This is how you PROVE that a change is an improvement, not just a change.

### Step 11: What happens if you swap A and B?

Good practice -- always check for position bias:

```python
result_swapped = compare_versions(task, version_b, version_a)
print(f"Swapped winner: Version {result_swapped['winner']}")
print(f"Confidence: {result_swapped['confidence']}")
```

If the winner flips just because you swapped positions, the quality difference is too small for the judge to reliably detect. For clear quality differences, the winner should stay the same regardless of position.

---

## Part 4: Building a Scoring Pipeline

Now let's combine everything into a single evaluation pipeline. Heuristics for structure, LLM-as-judge for quality, one combined score.

### Step 12: The evaluator class

```python
# 09-evaluation-fundamentals/step12_pipeline.py
import ollama
import json
import re


class TaskDescriptionEvaluator:
    """Combined heuristic + LLM evaluation pipeline."""

    def __init__(self, model="llama3.1:8b"):
        self.model = model

    def heuristic_score(self, text):
        """Fast, deterministic checks. Runs in microseconds."""
        checks = {
            "has_numbered_steps": len(
                re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)
            ) >= 3,
            "word_count_ok": 30 <= len(text.split()) <= 200,
            "has_safety_mention": any(
                w in text.lower()
                for w in ["ppe", "safety", "lockout", "gloves", "glasses"]
            ),
            "has_specific_refs": bool(
                re.search(r'[A-Z]{2,}-\d{2,}', text)
            ),
            "uses_active_voice": sum(
                1 for p in ["should be", "is to be", "must be done"]
                if p in text.lower()
            ) <= 1,
        }
        score = sum(checks.values()) / len(checks)
        return {"checks": checks, "score": round(score, 2)}

    def llm_score(self, text):
        """Deep quality assessment. Takes a few seconds."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rate this manufacturing task description 0-10 on: "
                        "clarity, completeness, safety, specificity, "
                        "professionalism. Return JSON with those keys "
                        "and an 'overall' key."
                    ),
                },
                {"role": "user", "content": text},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        scores = json.loads(response["message"]["content"])
        # Normalize to 0-1
        normalized = {}
        for key, val in scores.items():
            if isinstance(val, (int, float)):
                normalized[key] = round(val / 10, 2)
        return normalized

    def evaluate(self, text):
        """Full evaluation: heuristic + LLM, combined."""
        h = self.heuristic_score(text)
        l = self.llm_score(text)

        # Weighted: 30% structure, 70% quality
        combined = round(
            0.3 * h["score"] + 0.7 * l.get("overall", 0), 2
        )

        return {
            "heuristic": h,
            "llm": l,
            "combined_score": combined,
        }
```

### Step 13: Run it

```python
evaluator = TaskDescriptionEvaluator()

test_cases = [
    """CALIBRATE DIGITAL PRESSURE GAUGE

1. Ensure gauge is depressurized and at ambient temperature.
2. Connect gauge to calibration standard (NIST-traceable, +/-0.1%).
3. Apply test pressures at 0%, 25%, 50%, 75%, and 100% of range.
4. Record readings on Calibration Form CAL-201. Tolerance: +/-0.5%.
5. If out of tolerance, adjust per manufacturer instructions, re-test.
6. Apply calibration sticker with date and next-due date.""",

    "Check the pressure gauge. Make sure it reads right. Fix it if not.",
]

for i, desc in enumerate(test_cases):
    print(f"{'=' * 50}")
    print(f"Description {i + 1}:")
    result = evaluator.evaluate(desc)
    print(f"  Heuristic: {result['heuristic']['score']:.0%}")
    print(f"  LLM:       {result['llm'].get('overall', 'N/A')}")
    print(f"  Combined:  {result['combined_score']:.0%}")
    print()
```

### Step 14: Now imagine running this on 50 task descriptions

That's evaluation at scale:

```python
# Simulate batch evaluation
tasks_to_evaluate = [
    "Inspect incoming steel plate for surface defects per QC-101.",
    "Set up CNC mill for aluminum housing per drawing HG-2200-Rev.C.",
    "Perform daily forklift safety checklist per OSHA standard.",
    # In reality, you'd load these from your RAG pipeline output
]

print("=== Batch Evaluation Summary ===\n")
scores = []
for task in tasks_to_evaluate:
    # Use just heuristic for speed in batch mode
    result = evaluator.heuristic_score(task)
    scores.append(result["score"])
    status = "PASS" if result["score"] >= 0.6 else "FAIL"
    print(f"  [{status}] {result['score']:.0%} - {task[:50]}...")

avg = sum(scores) / len(scores)
print(f"\nAverage score: {avg:.0%}")
print(f"Pass rate (>= 60%): {sum(1 for s in scores if s >= 0.6)}/{len(scores)}")
```

---

## Part 5: Putting It All Together

### Step 15: The complete picture

Here is what you now have:

```
Evaluation Toolbox
==================

Layer 1: Heuristic (FREE, INSTANT)
  - Numbered steps? Format right?
  - Safety terms present?
  - Specific references included?
  -> Use for: every single output, gating bad ones early

Layer 2: LLM-as-Judge (COSTS COMPUTE, SECONDS)
  - Is it clear? Complete? Professional?
  - Does the quality match expectations?
  -> Use for: deeper assessment, periodic audits

Layer 3: A/B Comparison (COSTS COMPUTE, SECONDS)
  - Is this version better than that version?
  -> Use for: testing prompt changes, comparing models

Combined Pipeline:
  Heuristic score (30%) + LLM score (70%) = combined score
  -> Use for: stakeholder reporting
```

Now you can answer your boss:

> "We tested 50 task descriptions against our quality rubric. The AI scored 87% on format compliance and 92% on safety inclusion. The main gap is referencing specific form numbers -- I'm adding more reference documents to improve that."

That's not a vibe. That's data.

---

## Takeaways

1. **Start with heuristics** -- they are free, instant, and deterministic. Use them as your first line of defense.
2. **LLM-as-judge catches quality** that rules miss -- clarity, completeness, professionalism.
3. **A/B comparison proves improvements** -- don't guess, measure.
4. **Combine both layers** into a scoring pipeline for the full picture.
5. **Your evaluation rubric IS your quality standard** -- define what "good" means before you measure it.

## What's Next

You built evaluation from scratch. Module 10 introduces **Ragas** -- a purpose-built framework that does this specifically for RAG pipelines, with research-backed metrics like faithfulness and context precision. It does in a few lines what took us an entire module to build.
