# Module 09: Evaluation Fundamentals

## Goal
Learn the principles and techniques of LLM evaluation. This is **the most important skill** in this entire learning journey — it's what separates "I think the AI is doing OK" from "I can prove the AI output improved by 18% after this change."

---

## Concepts

### Why Evaluation Is Everything

Consider this conversation at work:

> **Manager:** "Can we trust the AI to write task descriptions?"
> **You (without evaluation):** "Yeah, they look pretty good to me."
> **You (with evaluation):** "We tested 50 task descriptions against our quality rubric. The AI scored 87% on format compliance, 92% on safety inclusion, and 78% on specificity. The main gap is referencing specific form numbers — I'm adding more reference documents to improve that."

Which person gets the project approved?

### The Evaluation Landscape

```
                    ┌─────────────────────────┐
                    │   EVALUATION METHODS     │
                    └────────────┬────────────┘
           ┌─────────────────────┼─────────────────────┐
           ↓                     ↓                     ↓
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Heuristic   │    │  LLM-as-     │    │   Human      │
    │  (Rule-based)│    │  Judge       │    │  Evaluation  │
    └──────────────┘    └──────────────┘    └──────────────┘
    - Word count        - Quality scoring   - Expert review
    - Format checks     - Comparison        - A/B testing
    - Regex patterns    - Rubric-based      - Preference ranking
    - Keyword presence  - Style matching    - Annotation

    Fast & cheap        Good balance        Gold standard
    Limited depth       Scalable            Expensive & slow
```

### Key Metrics for Your Use Case

| Metric | What It Measures | Why You Care |
|--------|-----------------|--------------|
| **Format compliance** | Does it match your template? | Consistency across all task descriptions |
| **Factual accuracy** | Are specs/form numbers correct? | Trust & safety |
| **Completeness** | Are all required sections present? | Regulatory compliance |
| **Readability** | Can operators understand it? | Practical usability |
| **Style consistency** | Does it match existing docs? | Professional appearance |
| **Safety inclusion** | Are safety warnings present? | Liability & worker safety |

---

## Exercise 1: Rule-Based (Heuristic) Evaluation

```python
# 09-evaluation-fundamentals/ex1_heuristic_eval.py
"""Build rule-based evaluation functions for task descriptions."""

import re
import json


def evaluate_task_description(description: str) -> dict:
    """Evaluate a task description against manufacturing quality rules."""
    scores = {}

    # 1. Format: Has numbered steps?
    numbered_steps = re.findall(r'^\s*\d+[\.\)]\s', description, re.MULTILINE)
    scores["has_numbered_steps"] = len(numbered_steps) >= 3

    # 2. Length: Reasonable word count?
    words = description.split()
    scores["appropriate_length"] = 30 <= len(words) <= 200
    scores["word_count"] = len(words)

    # 3. Action verbs: Steps start with verbs?
    action_verbs = ["inspect", "verify", "check", "install", "remove", "clean",
                    "record", "document", "apply", "ensure", "confirm", "test",
                    "measure", "adjust", "replace", "tighten", "loosen", "calibrate"]
    verb_count = sum(1 for verb in action_verbs if verb.lower() in description.lower())
    scores["uses_action_verbs"] = verb_count >= 2
    scores["action_verb_count"] = verb_count

    # 4. Safety mentions
    safety_terms = ["ppe", "safety", "glasses", "gloves", "helmet", "lockout",
                    "tagout", "caution", "warning", "hazard", "protective"]
    safety_count = sum(1 for term in safety_terms if term.lower() in description.lower())
    scores["mentions_safety"] = safety_count >= 1
    scores["safety_term_count"] = safety_count

    # 5. References specifics (form numbers, specs, part numbers)
    has_form_ref = bool(re.search(r'[A-Z]{2,}-\d{2,}', description))
    has_measurement = bool(re.search(r'\d+\.?\d*\s*(mm|cm|inch|"|Nm|PSI|°[FC]|CFH|RPM)', description))
    scores["has_references"] = has_form_ref or has_measurement
    scores["has_form_numbers"] = has_form_ref
    scores["has_measurements"] = has_measurement

    # 6. Active voice check (simple heuristic)
    passive_indicators = ["should be", "must be", "is to be", "are to be", "was done", "were performed"]
    passive_count = sum(1 for p in passive_indicators if p in description.lower())
    scores["mostly_active_voice"] = passive_count <= 1

    # 7. Overall score
    binary_checks = [
        scores["has_numbered_steps"],
        scores["appropriate_length"],
        scores["uses_action_verbs"],
        scores["mentions_safety"],
        scores["has_references"],
        scores["mostly_active_voice"],
    ]
    scores["overall_score"] = sum(binary_checks) / len(binary_checks)
    scores["overall_percent"] = f"{scores['overall_score'] * 100:.0f}%"

    return scores


# Test with different quality task descriptions
examples = {
    "good": """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete on the welding station before beginning inspection.
2. Don required PPE: safety glasses, leather gloves, and inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6 criteria — check for cracks, porosity, and undercut.
4. Measure weld size with fillet gauge — minimum 6mm leg per drawing SH-4402.
5. Record all findings on Form QC-107. Tag any defective joints with red HOLD tag and notify shift supervisor.""",

    "mediocre": """Weld Inspection

Look at the welds and make sure they are good. Check for any problems.
If something looks wrong, tell someone about it.
Make sure to write it down when you're done.""",

    "bad": "Check the welds.",
}

print("=== Heuristic Evaluation Results ===\n")
for quality, description in examples.items():
    scores = evaluate_task_description(description)
    print(f"--- {quality.upper()} Example ---")
    print(f"Overall: {scores['overall_percent']}")
    print(f"  Numbered steps: {'✓' if scores['has_numbered_steps'] else '✗'}")
    print(f"  Length ({scores['word_count']} words): {'✓' if scores['appropriate_length'] else '✗'}")
    print(f"  Action verbs ({scores['action_verb_count']}): {'✓' if scores['uses_action_verbs'] else '✗'}")
    print(f"  Safety mentions: {'✓' if scores['mentions_safety'] else '✗'}")
    print(f"  Specific references: {'✓' if scores['has_references'] else '✗'}")
    print(f"  Active voice: {'✓' if scores['mostly_active_voice'] else '✗'}")
    print()

print("=== When to Use Heuristic Evaluation ===")
print("✓ Fast (milliseconds), free, deterministic")
print("✓ Great for format/structure checks")
print("✗ Can't assess quality, coherence, or meaning")
print("✗ Brittle — 'wears PPE' passes but 'dons protective equipment' might not")
print("\nCombine with LLM-as-judge (Exercise 2) for deeper evaluation.")
```

---

## Exercise 2: LLM-as-Judge

```python
# 09-evaluation-fundamentals/ex2_llm_as_judge.py
"""Use an LLM to evaluate LLM output — the most scalable deep evaluation method."""

import ollama
import json

JUDGE_SYSTEM_PROMPT = """You are an expert manufacturing technical writing evaluator.
You evaluate task descriptions for manufacturing environments.

Score each task description on these criteria (0-10 scale):

1. CLARITY: Is it clear what the operator needs to do? (0=confusing, 10=crystal clear)
2. COMPLETENESS: Are all necessary steps included? (0=missing critical steps, 10=comprehensive)
3. SAFETY: Are relevant safety precautions mentioned? (0=dangerous omissions, 10=thorough safety coverage)
4. SPECIFICITY: Does it reference specific tools, specs, forms? (0=vague, 10=very specific)
5. PROFESSIONALISM: Does it read like a professional manufacturing document? (0=casual, 10=professional)

Return your evaluation as JSON:
{
  "clarity": {"score": <0-10>, "reasoning": "brief explanation"},
  "completeness": {"score": <0-10>, "reasoning": "brief explanation"},
  "safety": {"score": <0-10>, "reasoning": "brief explanation"},
  "specificity": {"score": <0-10>, "reasoning": "brief explanation"},
  "professionalism": {"score": <0-10>, "reasoning": "brief explanation"},
  "overall_score": <0-10 average>,
  "improvement_suggestions": ["suggestion 1", "suggestion 2"]
}"""


def judge_task_description(description: str, task_context: str = "") -> dict:
    """Have an LLM evaluate a task description."""
    user_prompt = f"Evaluate this manufacturing task description:\n\n{description}"
    if task_context:
        user_prompt += f"\n\nContext: This was written for the task: '{task_context}'"

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.0},
    )

    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {"error": "Failed to parse judge response"}


# Test with our same examples
examples = {
    "good": """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete before beginning inspection.
2. Don required PPE: safety glasses, leather gloves, inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6 — check for cracks, porosity, undercut.
4. Measure weld size with fillet gauge — minimum 6mm leg per drawing SH-4402.
5. Record findings on Form QC-107. Tag defective joints with red HOLD tag, notify supervisor.""",

    "mediocre": """Weld Inspection

Look at the welds and make sure they are good. Check for any problems.
If something looks wrong, tell someone about it.
Make sure to write it down when you're done.""",
}

print("=== LLM-as-Judge Evaluation ===\n")
for quality, description in examples.items():
    print(f"--- {quality.upper()} Example ---")
    evaluation = judge_task_description(
        description,
        task_context="Inspect welded joints on Frame Assembly A",
    )

    if "error" not in evaluation:
        for criterion in ["clarity", "completeness", "safety", "specificity", "professionalism"]:
            if criterion in evaluation:
                entry = evaluation[criterion]
                score = entry.get("score", "?")
                reason = entry.get("reasoning", "")
                bar = "█" * int(float(score)) if str(score).replace(".", "").isdigit() else ""
                print(f"  {criterion:15s}: {score}/10 {bar}")
                print(f"  {'':15s}  → {reason}")

        if "overall_score" in evaluation:
            print(f"\n  Overall: {evaluation['overall_score']}/10")
        if "improvement_suggestions" in evaluation:
            print(f"  Suggestions: {evaluation['improvement_suggestions']}")
    else:
        print(f"  Error: {evaluation['error']}")
    print()

print("=== LLM-as-Judge: Pros and Cons ===")
print("✓ Understands quality, coherence, and context")
print("✓ Can evaluate nuanced criteria (professionalism, clarity)")
print("✓ Scalable — can evaluate thousands of outputs")
print("✗ Non-deterministic — may give slightly different scores each run")
print("✗ Biased toward verbose outputs (they 'seem' better)")
print("✗ Costs compute (or money for cloud models)")
print("\nBest practice: Use heuristics for structure, LLM-as-judge for quality.")
```

---

## Exercise 3: Comparison-Based Evaluation (A/B Testing)

```python
# 09-evaluation-fundamentals/ex3_ab_testing.py
"""Compare two versions of output to determine which is better."""

import ollama
import json

COMPARISON_PROMPT = """You are comparing two manufacturing task descriptions.
Determine which is BETTER for actual use on a manufacturing floor.

Consider:
- Clarity for operators with varying reading levels
- Safety coverage
- Actionability (can an operator follow this?)
- Professional formatting

TASK: {task}

VERSION A:
{version_a}

VERSION B:
{version_b}

Return JSON:
{{
  "winner": "A" or "B",
  "confidence": "high" / "medium" / "low",
  "reasoning": "why the winner is better",
  "version_a_strengths": ["..."],
  "version_a_weaknesses": ["..."],
  "version_b_strengths": ["..."],
  "version_b_weaknesses": ["..."]
}}"""


def compare_versions(task: str, version_a: str, version_b: str) -> dict:
    """A/B test two task description versions using LLM judgment."""
    prompt = COMPARISON_PROMPT.format(
        task=task, version_a=version_a, version_b=version_b
    )

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0},
    )

    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {"error": "Parse failed"}


# Compare: RAG-enhanced vs. plain LLM output
task = "Replace worn conveyor belt rollers"

version_a = """Replace Conveyor Belt Rollers

1. Lock out conveyor system per LOTO procedure SOP-SAFE-001.
2. Don PPE: safety glasses, leather gloves, steel-toe boots.
3. Remove belt tension using the take-up adjustment (refer to manual Section 3.2).
4. Remove worn rollers and inspect bearing seats for damage.
5. Install new rollers (P/N CR-200 series), verify free rotation.
6. Restore belt tension to 2-3% elongation per specification.
7. Run conveyor empty for 5 minutes, check tracking and roller alignment.
8. Document replacement on Form PM-105 and update maintenance log."""

version_b = """Change the old rollers on the conveyor belt.

Turn off the conveyor first. Take out the bad rollers and put in new ones.
Make sure the belt works when you turn it back on.
Write down what you did."""

result = compare_versions(task, version_a, version_b)
print("=== A/B Comparison Result ===\n")
print(f"Winner: Version {result.get('winner', '?')}")
print(f"Confidence: {result.get('confidence', '?')}")
print(f"Reasoning: {result.get('reasoning', '?')}")
print(f"\nVersion A strengths: {result.get('version_a_strengths', [])}")
print(f"Version A weaknesses: {result.get('version_a_weaknesses', [])}")
print(f"Version B strengths: {result.get('version_b_strengths', [])}")
print(f"Version B weaknesses: {result.get('version_b_weaknesses', [])}")

print("\n=== A/B Testing in Practice ===")
print("Use this to test:")
print("  - Prompt version A vs B")
print("  - RAG vs no-RAG output")
print("  - Different models on the same task")
print("  - Before/after adding new reference documents")
print("\nThis is how you PROVE that changes make things better (or worse).")
```

---

## Exercise 4: Building a Scoring Pipeline

```python
# 09-evaluation-fundamentals/ex4_scoring_pipeline.py
"""Combine heuristic and LLM evaluation into a single pipeline."""

import ollama
import json
import re


class TaskDescriptionEvaluator:
    """Combined heuristic + LLM evaluation pipeline."""

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def heuristic_score(self, text: str) -> dict:
        """Fast, deterministic checks."""
        checks = {
            "has_numbered_steps": bool(re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)),
            "word_count_ok": 30 <= len(text.split()) <= 200,
            "has_safety_mention": any(
                w in text.lower()
                for w in ["ppe", "safety", "lockout", "gloves", "glasses", "caution"]
            ),
            "has_specific_refs": bool(re.search(r'[A-Z]{2,}-\d{2,}', text)),
            "has_measurements": bool(re.search(r'\d+\.?\d*\s*(mm|Nm|PSI|°)', text)),
            "uses_active_voice": sum(
                1 for p in ["should be", "is to be", "must be done"]
                if p in text.lower()
            ) <= 1,
        }
        checks["heuristic_score"] = sum(checks.values()) / len(checks)
        return checks

    def llm_score(self, text: str) -> dict:
        """Deep quality assessment using LLM judge."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Rate this manufacturing task description 0-10 on: "
                    "clarity, completeness, safety, specificity, professionalism. "
                    "Return JSON: {\"clarity\": N, \"completeness\": N, \"safety\": N, "
                    "\"specificity\": N, \"professionalism\": N, \"overall\": N}",
                },
                {"role": "user", "content": text},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        try:
            scores = json.loads(response["message"]["content"])
            # Normalize to 0-1
            for key in scores:
                if isinstance(scores[key], (int, float)):
                    scores[key] = round(scores[key] / 10, 2)
            return scores
        except json.JSONDecodeError:
            return {"error": "parse_failed"}

    def evaluate(self, text: str) -> dict:
        """Full evaluation: heuristic + LLM."""
        heuristic = self.heuristic_score(text)
        llm = self.llm_score(text)

        # Combined score (weighted average)
        h_score = heuristic.get("heuristic_score", 0)
        l_score = llm.get("overall", 0)
        combined = round(0.3 * h_score + 0.7 * l_score, 2)

        return {
            "heuristic": heuristic,
            "llm": llm,
            "combined_score": combined,
        }


# Use it
evaluator = TaskDescriptionEvaluator()

test_descriptions = [
    """CALIBRATE DIGITAL PRESSURE GAUGE

1. Ensure gauge is depressurized and at ambient temperature.
2. Connect gauge to calibration standard (NIST-traceable, ±0.1% accuracy).
3. Apply test pressures at 0%, 25%, 50%, 75%, and 100% of range.
4. Record readings on Calibration Form CAL-201. Tolerance: ±0.5% of full scale.
5. If out of tolerance, adjust per manufacturer instructions and re-test.
6. Apply calibration sticker with date and next-due date. Return to service.""",

    """Check the pressure gauge. Make sure it reads right. Fix it if it doesn't.""",
]

for i, desc in enumerate(test_descriptions):
    print(f"{'='*60}")
    print(f"Description {i+1}:")
    print(f"{'='*60}")
    result = evaluator.evaluate(desc)
    print(f"\nHeuristic score: {result['heuristic']['heuristic_score']:.0%}")
    print(f"LLM score: {result['llm'].get('overall', 'N/A')}")
    print(f"Combined score: {result['combined_score']:.0%}")
    print()
```

---

## Takeaways

1. **Evaluation has three pillars**: heuristic (fast/cheap), LLM-as-judge (balanced), human review (gold standard)
2. **Heuristics catch structure issues**, LLM-as-judge catches quality issues — use both
3. **A/B testing** is how you prove changes are improvements, not just changes
4. **Build evaluation into your pipeline**, not as an afterthought
5. **Your evaluation rubric IS your quality standard** — define what "good" means before measuring it

## Setting the Stage for Module 10

You built custom evaluation from scratch. Module 10 introduces **Ragas** — a purpose-built framework for evaluating RAG pipelines with research-backed metrics like faithfulness, answer relevancy, and context precision. It does in a few lines what took us an entire module to build.
