# Module 09: Evaluation Fundamentals

## Goal
Answer the question every stakeholder will ask: **"How do we know the AI is doing a good job?"** By the end of this module, you will have a concrete, numbers-backed answer.

---

## The Scene

Your boss walks into your office.

> **Boss:** "So this AI thing you've been building for scouting reports... how do we know it's actually good?"
>
> **You (today):** "Uh, they look pretty good to me?"
>
> **You (after this module):** "We tested 50 scouting reports against our quality rubric. The AI scored 87% on format compliance, 92% on stat inclusion, and 78% on specificity. The main gap is referencing specific comparable players -- I'm adding more reference docs to close that gap."

Which person gets the project approved?

Let's build that answer. Start simple.

---

## Part 1: Rule-Based Checks (Free and Instant)

Before anything fancy, let's ask the simplest possible question: does the AI output even *look* right?

### Step 1: One tiny check

Open a Python shell or create a new file:

```python
# 09-evaluation-fundamentals/step1_one_check.py

good_report = """SCOUTING REPORT: QB PROSPECT – MARCUS ALLEN

1. Pocket passer with elite accuracy, completing 68% of passes with a 2.3-second average release.
2. Excels on intermediate routes (15-25 yards), consistently hitting tight windows.
3. Reads defenses pre-snap and adjusts protection calls at the line.
4. Arm strength measured at 62 mph at the combine.
5. Weakness: locks onto first read under pressure, leading to forced throws."""

bad_report = "He looks like a good quarterback."
```

Run this:

```python
import re

def has_numbered_steps(text):
    steps = re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)
    return len(steps)

print(has_numbered_steps(good_report))  # What number do you see?
print(has_numbered_steps(bad_report))   # And this one?
```

You should see `5` and `0`. That single function just told you something useful: one output has structure, the other doesn't. And it ran in microseconds. No LLM needed.

### Step 2: Check for measurables

Now let's add another check. In football scouting, measurables and stats are non-negotiable:

```python
def count_measurable_mentions(text):
    measurable_terms = [
        "40-yard", "speed", "vertical", "arm strength",
        "completion", "yards", "combine", "measurables",
        "release", "dash", "broad jump", "shuttle"
    ]
    text_lower = text.lower()
    return sum(1 for term in measurable_terms if term in text_lower)

print(count_measurable_mentions(good_report))  # Try it
print(count_measurable_mentions(bad_report))   # And this
```

The good report should score 4+ (speed, arm strength, completion, release). The bad report? Zero. You now have two checks that cost nothing and run instantly.

### Step 3: Action verbs

Scouting reports should use descriptive, analytical language -- "excels", "demonstrates", "projects" -- not vague language like "seems okay":

```python
def count_action_verbs(text):
    verbs = [
        "excels", "demonstrates", "projects", "reads",
        "completes", "runs", "catches", "blocks",
        "covers", "tracks", "attacks", "diagnoses",
        "exhibits", "displays", "measures", "ranks",
        "adjusts", "anticipates", "delivers", "accelerates"
    ]
    text_lower = text.lower()
    return sum(1 for v in verbs if v in text_lower)

print(count_action_verbs(good_report))
print(count_action_verbs(bad_report))
```

Notice how each check is just a few lines. Nothing complicated. But together they start to paint a picture.

### Step 4: Specific references

Real scouting reports cite stats, measurables, and comparable players. Vague ones don't:

```python
def has_specific_references(text):
    has_player_ref = bool(re.search(r'[A-Z]{2,}-\d{2,}', text))
    has_measurement = bool(re.search(
        r'\d+\.?\d*\s*(mph|yards|%|inch|seconds|reps|"|vertical)', text
    ))
    return has_player_ref, has_measurement

form, meas = has_specific_references(good_report)
print(f"Report IDs: {form}, Measurables: {meas}")

form, meas = has_specific_references(bad_report)
print(f"Report IDs: {form}, Measurables: {meas}")
```

The good report has measurables (68%, 2.3-second, 62 mph, 15-25 yards). The bad report has neither.

### Step 5: Combine them into a score

Now let's put it all together. Each check becomes a pass/fail, and we get a percentage:

```python
def heuristic_score(text):
    checks = {
        "has_numbered_steps": has_numbered_steps(text) >= 3,
        "appropriate_length": 30 <= len(text.split()) <= 200,
        "uses_action_verbs": count_action_verbs(text) >= 2,
        "mentions_measurables": count_measurable_mentions(text) >= 1,
        "has_report_ids": has_specific_references(text)[0],
        "has_stats": has_specific_references(text)[1],
    }
    score = sum(checks.values()) / len(checks)
    return checks, score

checks, score = heuristic_score(good_report)
print(f"Good report: {score:.0%}")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")

print()

checks, score = heuristic_score(bad_report)
print(f"Bad report: {score:.0%}")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")
```

You should see something like 100% vs 0%. That's a clear signal, and it took zero LLM calls.

### Step 6: Test the middle ground

What about a "mediocre" scouting report? Let's see if our scoring catches it:

```python
mediocre_report = """Quarterback Evaluation

He can throw pretty well and seems athletic. He has a decent arm.
If he gets more experience, he could be good.
Probably worth a look in the later rounds."""

checks, score = heuristic_score(mediocre_report)
print(f"Mediocre report: {score:.0%}")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")
```

Notice what happens: it passes on length and maybe some verbs, but fails on numbered steps, measurables, and references. It correctly lands between good and bad.

**Key insight:** Rules catch structure but they can NOT catch quality. "He runs fast. He catches well. He blocks okay." would pass some checks, but it's still a terrible scouting report. We need something smarter.

---

## Part 2: LLM-as-Judge

The idea: use an LLM to evaluate another LLM's output. The judge LLM reads the scouting report and scores it, just like a senior analyst would -- but at machine speed.

### Step 7: Score one thing -- clarity

Let's start with a single criterion. Keep it focused:

```python
# 09-evaluation-fundamentals/step7_llm_judge.py
from openai import OpenAI
import json

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

def judge_clarity(text):
    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate NFL scouting reports. "
                    "Score the CLARITY on a 0-10 scale. "
                    "0 = confusing, 10 = crystal clear for any scout or analyst. "
                    'Return JSON: {"clarity": <score>, "reason": "<why>"}'
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)
```

Try it on your good report:

```python
result = judge_clarity(good_report)
print(f"Clarity: {result['clarity']}/10")
print(f"Reason: {result['reason']}")
```

Now the bad report:

```python
result = judge_clarity(bad_report)
print(f"Clarity: {result['clarity']}/10")
print(f"Reason: {result['reason']}")
```

See the difference? The LLM can judge *meaning*, not just structure. It understands that "He looks like a good quarterback" is vague in a way that regex never could.

### Step 8: Add more criteria

Now let's expand to a full rubric. Notice how we build on the same pattern:

```python
JUDGE_PROMPT = """You evaluate NFL scouting reports.
Score on each criterion (0-10):

1. CLARITY: Can any scout or analyst understand the evaluation?
2. COMPLETENESS: Are all key areas covered (strengths, weaknesses, measurables)?
3. STAT_SUPPORT: Are claims backed by specific stats and measurables?
4. SPECIFICITY: Does it cite measurables, game film details, comparable players?
5. PROFESSIONALISM: Does it read like a real NFL scouting report?

Return JSON:
{
  "clarity": {"score": N, "reason": "..."},
  "completeness": {"score": N, "reason": "..."},
  "stat_support": {"score": N, "reason": "..."},
  "specificity": {"score": N, "reason": "..."},
  "professionalism": {"score": N, "reason": "..."},
  "overall": N
}"""

def judge_full(text):
    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)
```

Run it:

```python
result = judge_full(good_report)
for criterion in ["clarity", "completeness", "stat_support", "specificity", "professionalism"]:
    entry = result.get(criterion, {})
    score = entry.get("score", "?")
    reason = entry.get("reason", "")
    bar = "#" * int(score) if isinstance(score, (int, float)) else ""
    print(f"  {criterion:15s}: {score}/10  {bar}")
    print(f"  {'':15s}  -> {reason}")
print(f"\n  Overall: {result.get('overall', '?')}/10")
```

Now run the same thing on the mediocre report:

```python
result = judge_full(mediocre_report)
for criterion in ["clarity", "completeness", "stat_support", "specificity", "professionalism"]:
    entry = result.get(criterion, {})
    score = entry.get("score", "?")
    reason = entry.get("reason", "")
    print(f"  {criterion:15s}: {score}/10 -> {reason}")
```

Compare the scores side by side. The LLM judge catches nuance that rules miss: it knows "seems athletic" is unprofessional, and it knows the mediocre version omits critical measurables and comparable player analysis.

### Step 9: Watch out for bias

One gotcha -- LLM judges tend to prefer longer, more verbose outputs. Let's see:

```python
verbose_but_bad = """COMPREHENSIVE NFL DRAFT PROSPECT EVALUATION REPORT

1. Prior to commencing any evaluation activities, it is imperative that the
   designated scouting personnel verify and confirm that all applicable
   game film has been properly and thoroughly reviewed and catalogued.
2. The evaluation analyst shall document all requisite athletic
   measurements including but not limited to forty-yard dash times, vertical
   leap measurements, and broad jump distances.
3. A comprehensive physical assessment shall be conducted."""

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
from openai import OpenAI
import json

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

COMPARE_PROMPT ="""You are comparing two NFL scouting reports for the same player.
Which is BETTER for actual use by a front office or coaching staff?

Consider:
- Can a scout or GM actually use it to make draft decisions?
- Are measurables and stats included?
- Are specific strengths and weaknesses identified?

PLAYER: {task}

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
    response = llm.chat.completions.create(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)
```

Let's test it -- RAG-enhanced vs. plain LLM output:

```python
task = "Evaluate RB prospect Jaylen Carter"

version_a = """Scouting Report: RB Jaylen Carter

1. Explosive runner with 4.38 40-yard dash and elite acceleration through the hole.
2. Exceptional vision -- finds cutback lanes and hits them decisively.
3. Averaging 3.8 yards after contact, top-10 among draft-eligible backs.
4. 45 receptions out of the backfield, showing reliable hands on check-downs.
5. Weakness: pass protection needs significant work -- struggles with blitz pickup.
6. Comparable players: early-career Alvin Kamara (receiving) with Kareem Hunt's power.
7. Projection: late first to early second round. Day-one starter potential.
8. Recommend: draft if available in Round 2, per draft board DB-2025."""

version_b = """He's a fast running back who runs hard.

He can catch the ball sometimes. He needs to get better at blocking.
He would probably be a good pick at some point in the draft."""

result = compare_versions(task, version_a, version_b)
print(f"Winner: Version {result['winner']}")
print(f"Confidence: {result['confidence']}")
print(f"Key difference: {result['key_difference']}")
```

No surprise which wins. But notice the power here: you can use this to compare prompt versions, before/after RAG improvements, or different models on the same player. This is how you PROVE that a change is an improvement, not just a change.

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
from openai import OpenAI
import json
import re


class ScoutingReportEvaluator:
    """Combined heuristic + LLM evaluation pipeline."""

    def __init__(self, model="gemma3:12b"):
        self.model = model
        self.llm = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def heuristic_score(self, text):
        """Fast, deterministic checks. Runs in microseconds."""
        checks = {
            "has_numbered_steps": len(
                re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)
            ) >= 3,
            "word_count_ok": 30 <= len(text.split()) <= 200,
            "has_measurables": any(
                w in text.lower()
                for w in ["40-yard", "speed", "vertical", "arm strength", "combine"]
            ),
            "has_specific_refs": bool(
                re.search(r'[A-Z]{2,}-\d{2,}', text)
            ),
            "mostly_active_voice": sum(
                1 for p in ["should be", "is to be", "must be done"]
                if p in text.lower()
            ) <= 1,
        }
        score = sum(checks.values()) / len(checks)
        return {"checks": checks, "score": round(score, 2)}

    def llm_score(self, text):
        """Deep quality assessment. Takes a few seconds."""
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rate this NFL scouting report 0-10 on: "
                        "clarity, completeness, stat_support, specificity, "
                        "professionalism. Return JSON with those keys "
                        "and an 'overall' key."
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        scores = json.loads(response.choices[0].message.content)
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
evaluator = ScoutingReportEvaluator()

test_cases = [
    """SCOUTING REPORT: OL PROSPECT – DAVID THOMPSON

1. Excellent pass protection anchor with quick lateral movement and 34-inch arms.
2. Run blocking grade: 82.5/100 per PFF, excels at drive blocks and reach blocks.
3. Allowed only 2 sacks in 580 pass-blocking snaps last season.
4. Shows elite football IQ, rarely makes assignment errors per film review SR-4401.
5. Weakness: struggles with combo blocks at the second level.
6. Projection: Round 1-2, immediate starter at guard or right tackle.""",

    "He's a big guy who blocks well. Probably a good lineman.",
]

for i, desc in enumerate(test_cases):
    print(f"{'=' * 50}")
    print(f"Report {i + 1}:")
    result = evaluator.evaluate(desc)
    print(f"  Heuristic: {result['heuristic']['score']:.0%}")
    print(f"  LLM:       {result['llm'].get('overall', 'N/A')}")
    print(f"  Combined:  {result['combined_score']:.0%}")
    print()
```

### Step 14: Now imagine running this on 50 scouting reports

That's evaluation at scale:

```python
# Simulate batch evaluation
reports_to_evaluate = [
    "Pocket passer with elite accuracy, 68% completion rate per report QB-101.",
    "Explosive runner with 4.38 40-yard dash, exceptional vision per RB-201.",
    "Crisp route runner with 4.42 speed, 38-inch vertical per WR-301.",
    # In reality, you'd load these from your RAG pipeline output
]

print("=== Batch Evaluation Summary ===\n")
scores = []
for report in reports_to_evaluate:
    # Use just heuristic for speed in batch mode
    result = evaluator.heuristic_score(report)
    scores.append(result["score"])
    status = "PASS" if result["score"] >= 0.6 else "FAIL"
    print(f"  [{status}] {result['score']:.0%} - {report[:50]}...")

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
  - Measurables and stats present?
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

> "We tested 50 scouting reports against our quality rubric. The AI scored 87% on format compliance and 92% on stat inclusion. The main gap is referencing comparable players -- I'm adding more reference documents to improve that."

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
