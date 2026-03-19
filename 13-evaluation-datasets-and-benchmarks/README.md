# Module 13: Evaluation Datasets & Benchmarks

## Goal
Build high-quality evaluation datasets and regression benchmarks. This is the capstone of Phase 3 — after this module, you'll have a complete evaluation infrastructure.

---

## Concepts

### The Evaluation Data Hierarchy

```
                ┌───────────────────┐
                │   Golden Dataset  │  ← Human-curated, expert-reviewed
                │   (20-50 examples)│     The source of truth
                └────────┬──────────┘
                         │
                ┌────────┴──────────┐
                │  Synthetic Dataset│  ← LLM-generated, human-validated
                │  (200-500 examples│     Broad coverage
                └────────┬──────────┘
                         │
                ┌────────┴──────────┐
                │ Production Traces │  ← Real user queries + feedback
                │  (continuous)     │     The real world
                └───────────────────┘
```

### What Makes a Good Evaluation Dataset?

1. **Representative** — covers the actual tasks your system handles
2. **Diverse** — includes easy, medium, and hard cases
3. **Labeled** — has expected answers or quality scores
4. **Versioned** — tracked in git, updated intentionally
5. **Adversarial** — includes edge cases and expected failures

---

## Exercise 1: Building a Golden Dataset

```python
# 13-evaluation-datasets-and-benchmarks/ex1_golden_dataset.py
"""Create a human-curated golden dataset for manufacturing task descriptions."""

import json

# Your golden dataset: expert-written examples with quality annotations
GOLDEN_DATASET = [
    {
        "id": "gold-001",
        "task_name": "Inspect welded joints on Frame Assembly A",
        "department": "Quality Control",
        "context": "AWS D1.1, Form QC-107, fillet gauge required",
        "expected_output": """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete on the welding station.
2. Don required PPE: safety glasses, leather gloves, inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6 — check for cracks, porosity, and undercut.
4. Measure weld size with fillet gauge — minimum 6mm leg per drawing.
5. Record findings on Form QC-107. Tag defective joints with red HOLD tag and notify supervisor.""",
        "quality_scores": {
            "format": 1.0,
            "safety": 1.0,
            "specificity": 0.9,
            "completeness": 0.9,
            "professionalism": 1.0,
        },
        "required_elements": ["numbered_steps", "ppe", "specification_reference", "form_reference", "action_verbs"],
        "difficulty": "medium",
        "tags": ["welding", "quality", "inspection"],
    },
    {
        "id": "gold-002",
        "task_name": "Perform daily forklift pre-operation inspection",
        "department": "Warehouse",
        "context": "OSHA 1910.178, Company SOP-FL-001",
        "expected_output": """DAILY FORKLIFT PRE-OPERATION INSPECTION

1. Check tire condition and inflation pressure visually.
2. Test horn, headlights, backup alarm, and strobe light.
3. Verify hydraulic fluid level — add if below MIN mark on dipstick.
4. Inspect mast chains for wear, damage, or excessive slack.
5. Test service brake and parking brake before loading.
6. Record inspection results on daily checklist per SOP-FL-001. Do not operate if any item fails — report to maintenance.""",
        "quality_scores": {
            "format": 1.0,
            "safety": 0.8,
            "specificity": 0.9,
            "completeness": 1.0,
            "professionalism": 1.0,
        },
        "required_elements": ["numbered_steps", "specification_reference", "action_verbs", "fail_criteria"],
        "difficulty": "easy",
        "tags": ["warehouse", "safety", "daily_check"],
    },
    {
        "id": "gold-003",
        "task_name": "Set up CNC lathe for precision shaft machining",
        "department": "Machining",
        "context": "Drawing SH-4402-Rev.B, tolerance ±0.005\", Haas ST-20",
        "expected_output": """SET UP CNC LATHE FOR PRECISION SHAFT MACHINING

1. Review drawing SH-4402-Rev.B and verify material is staged (confirm heat lot).
2. Load program from DNC server — verify program number matches setup sheet.
3. Install 3-jaw chuck with soft jaws bored to part diameter +0.002".
4. Set tool offsets using the Renishaw tool setter. Verify against setup sheet (±0.001").
5. Run first article at 50% rapid, 75% feed override. Measure OD, length, and runout.
6. Record first article measurements on FAIR form. Proceed to production after QC approval.""",
        "quality_scores": {
            "format": 1.0,
            "safety": 0.5,
            "specificity": 1.0,
            "completeness": 1.0,
            "professionalism": 1.0,
        },
        "required_elements": ["numbered_steps", "drawing_reference", "measurements", "action_verbs", "first_article"],
        "difficulty": "hard",
        "tags": ["machining", "cnc", "precision"],
    },
    {
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
        "quality_scores": {
            "format": 1.0,
            "safety": 1.0,
            "specificity": 1.0,
            "completeness": 1.0,
            "professionalism": 1.0,
        },
        "required_elements": ["numbered_steps", "loto", "part_numbers", "measurements", "pressure_test"],
        "difficulty": "hard",
        "tags": ["maintenance", "hydraulics", "safety_critical"],
    },
    {
        "id": "gold-005",
        "task_name": "Calibrate digital caliper",
        "department": "Metrology",
        "context": "NIST-traceable gauge blocks, Calibration SOP-CAL-003, Form CAL-201",
        "expected_output": """CALIBRATE DIGITAL CALIPER

1. Clean caliper jaws and gauge blocks with lint-free cloth and isopropyl alcohol.
2. Zero the caliper with jaws fully closed — verify display reads 0.000".
3. Measure gauge blocks at 0.500", 1.000", 2.000", and 4.000" (NIST-traceable set).
4. Record all readings on Form CAL-201. Tolerance: ±0.001" at each point.
5. If any reading is out of tolerance, adjust per manufacturer instructions and re-test.
6. Apply calibration sticker with date, technician ID, and next-due date. Return to service.""",
        "quality_scores": {
            "format": 1.0,
            "safety": 0.3,
            "specificity": 1.0,
            "completeness": 1.0,
            "professionalism": 1.0,
        },
        "required_elements": ["numbered_steps", "form_reference", "measurements", "tolerance", "calibration_sticker"],
        "difficulty": "medium",
        "tags": ["metrology", "calibration", "quality"],
    },
]

# Save the golden dataset
output_path = "13-evaluation-datasets-and-benchmarks/golden_dataset.json"
with open(output_path, "w") as f:
    json.dump(GOLDEN_DATASET, f, indent=2)

print(f"✓ Golden dataset saved: {output_path}")
print(f"  {len(GOLDEN_DATASET)} examples")
print(f"  Difficulty distribution: {[d['difficulty'] for d in GOLDEN_DATASET]}")
print(f"  Departments: {list(set(d['department'] for d in GOLDEN_DATASET))}")

# Dataset statistics
avg_words = sum(len(d["expected_output"].split()) for d in GOLDEN_DATASET) / len(GOLDEN_DATASET)
print(f"  Avg expected length: {avg_words:.0f} words")

print("\n=== Golden Dataset Best Practices ===")
print("1. Have domain experts write/review expected outputs")
print("2. Include difficulty levels (easy/medium/hard)")
print("3. Tag with categories for filtered evaluation")
print("4. Include quality score breakdowns, not just pass/fail")
print("5. Version in git — treat changes like code changes")
print("6. Start with 20-50 examples, grow as you find edge cases")
```

---

## Exercise 2: Synthetic Dataset Generation

```python
# 13-evaluation-datasets-and-benchmarks/ex2_synthetic_dataset.py
"""Generate a larger synthetic dataset using LLM, validated against golden examples."""

import ollama
import json
import re

# Load golden dataset as reference
with open("13-evaluation-datasets-and-benchmarks/golden_dataset.json") as f:
    golden = json.load(f)

# Generate synthetic test cases
GENERATION_PROMPT = """Based on these example manufacturing task descriptions, generate 5 NEW,
DIFFERENT task descriptions with their metadata.

EXAMPLES OF THE FORMAT:
{examples}

Generate 5 new tasks covering different departments (assembly, painting, shipping,
electrical, quality lab). Return as a JSON array where each item has:
- task_name: string
- department: string
- context: string (relevant specs/forms)
- difficulty: "easy", "medium", or "hard"
- required_elements: list of what a good description should include

Return ONLY the JSON array."""

# Use golden examples as few-shot reference
examples = "\n\n".join(
    f"Task: {g['task_name']}\nDept: {g['department']}\nContext: {g['context']}"
    for g in golden[:3]
)

print("=== Generating Synthetic Test Cases ===\n")
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "user", "content": GENERATION_PROMPT.format(examples=examples)},
    ],
    format="json",
    options={"temperature": 0.7},
)

try:
    synthetic_tasks = json.loads(response["message"]["content"])
    if isinstance(synthetic_tasks, dict):
        # Handle if wrapped in a key
        for key in synthetic_tasks:
            if isinstance(synthetic_tasks[key], list):
                synthetic_tasks = synthetic_tasks[key]
                break

    print(f"Generated {len(synthetic_tasks)} synthetic test cases:\n")
    for task in synthetic_tasks:
        print(f"  [{task.get('difficulty', '?'):6s}] {task.get('task_name', '?')}")
        print(f"          Dept: {task.get('department', '?')}")
        print(f"          Context: {task.get('context', '?')[:60]}...")
        print()

    # Save synthetic dataset
    output_path = "13-evaluation-datasets-and-benchmarks/synthetic_dataset.json"
    with open(output_path, "w") as f:
        json.dump(synthetic_tasks, f, indent=2)
    print(f"✓ Saved to {output_path}")

except json.JSONDecodeError as e:
    print(f"Failed to parse: {e}")
    print("Raw response:", response["message"]["content"][:300])

print("\n=== Synthetic Dataset Workflow ===")
print("1. Generate with LLM (this exercise)")
print("2. Human review — remove bad examples, fix errors")
print("3. Generate expected outputs for each task")
print("4. Score expected outputs using golden dataset rubric")
print("5. Add to your evaluation pipeline")
print("\nSynthetic data is a STARTING POINT, not a replacement for expert review.")
```

---

## Exercise 3: Regression Benchmark Suite

```python
# 13-evaluation-datasets-and-benchmarks/ex3_regression_benchmark.py
"""Build a regression benchmark that catches quality degradation."""

import ollama
import json
import re
from datetime import datetime


class RegressionBenchmark:
    """Run a standardized benchmark and compare against baseline."""

    def __init__(self, golden_dataset_path: str):
        with open(golden_dataset_path) as f:
            self.golden = json.load(f)
        self.results_history = []

    def evaluate_single(self, task: dict, generated: str) -> dict:
        """Evaluate a single generated description against golden standard."""
        scores = {}

        # Format checks
        steps = re.findall(r'^\s*\d+[\.\)]', generated, re.MULTILINE)
        scores["has_steps"] = len(steps) >= 3

        # Length check
        word_count = len(generated.split())
        scores["appropriate_length"] = 30 <= word_count <= 200

        # Required elements
        required = task.get("required_elements", [])
        element_checks = {
            "numbered_steps": len(steps) >= 3,
            "ppe": any(w in generated.lower() for w in ["ppe", "glasses", "gloves", "helmet", "boots"]),
            "specification_reference": bool(re.search(r'[A-Z]{2,}[\s-]\d', generated)),
            "form_reference": bool(re.search(r'[Ff]orm\s+[A-Z]', generated)),
            "action_verbs": any(w in generated.lower() for w in
                ["inspect", "verify", "check", "install", "remove", "record", "measure"]),
            "loto": any(w in generated.lower() for w in ["lockout", "tagout", "loto"]),
            "measurements": bool(re.search(r'\d+\.?\d*\s*(mm|Nm|"|inch|PSI|°)', generated)),
            "drawing_reference": bool(re.search(r'[Dd]rawing', generated)),
            "fail_criteria": any(w in generated.lower() for w in ["fail", "reject", "hold", "do not"]),
            "part_numbers": bool(re.search(r'[A-Z]{1,3}-\d{3,}', generated)),
            "tolerance": bool(re.search(r'±', generated)),
            "first_article": "first article" in generated.lower() or "first part" in generated.lower(),
            "calibration_sticker": "calibration" in generated.lower() and "sticker" in generated.lower(),
            "pressure_test": "pressure" in generated.lower() and "test" in generated.lower(),
        }

        required_passed = sum(1 for r in required if element_checks.get(r, False))
        scores["required_elements"] = required_passed / max(len(required), 1)

        # Overall
        all_binary = [scores["has_steps"], scores["appropriate_length"]]
        scores["overall"] = (sum(all_binary) / len(all_binary) + scores["required_elements"]) / 2

        return scores

    def run_benchmark(self, model: str = "llama3.1:8b", system_prompt: str = None) -> dict:
        """Run the full benchmark suite."""
        if system_prompt is None:
            system_prompt = """You are a manufacturing technical writer.
Write task descriptions with numbered steps, safety requirements, and specific references."""

        results = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "scores": [],
            "summary": {},
        }

        for task in self.golden:
            # Generate
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Task: {task['task_name']}\nContext: {task['context']}"},
                ],
                options={"temperature": 0.0},
            )
            generated = response["message"]["content"]

            # Evaluate
            scores = self.evaluate_single(task, generated)
            scores["task_id"] = task["id"]
            scores["task_name"] = task["task_name"]
            results["scores"].append(scores)

        # Calculate summary
        all_overalls = [s["overall"] for s in results["scores"]]
        results["summary"] = {
            "avg_overall": sum(all_overalls) / len(all_overalls),
            "min_overall": min(all_overalls),
            "max_overall": max(all_overalls),
            "total_tasks": len(results["scores"]),
            "passing_tasks": sum(1 for s in all_overalls if s >= 0.7),
        }

        self.results_history.append(results)
        return results

    def compare_runs(self, run_a: dict, run_b: dict) -> dict:
        """Compare two benchmark runs."""
        delta = run_b["summary"]["avg_overall"] - run_a["summary"]["avg_overall"]
        return {
            "avg_overall_a": run_a["summary"]["avg_overall"],
            "avg_overall_b": run_b["summary"]["avg_overall"],
            "delta": delta,
            "improved": delta > 0,
            "significant": abs(delta) > 0.05,  # 5% threshold
        }


# Run the benchmark
benchmark = RegressionBenchmark("13-evaluation-datasets-and-benchmarks/golden_dataset.json")

# Run 1: Basic prompt
print("=== Benchmark Run 1: Basic Prompt ===")
run1 = benchmark.run_benchmark(
    system_prompt="Write a manufacturing task description."
)
print(f"Average: {run1['summary']['avg_overall']:.2%}")
print(f"Passing: {run1['summary']['passing_tasks']}/{run1['summary']['total_tasks']}")

# Run 2: Detailed prompt
print("\n=== Benchmark Run 2: Detailed Prompt ===")
run2 = benchmark.run_benchmark(
    system_prompt="""You are a senior manufacturing technical writer at an ISO 9001 facility.
Write task descriptions with:
- Numbered steps (3-7 steps), each starting with an action verb
- PPE/safety requirements where applicable
- References to specific forms, specifications, and part numbers
- Measurable acceptance criteria
- Active voice, 8th-grade reading level"""
)
print(f"Average: {run2['summary']['avg_overall']:.2%}")
print(f"Passing: {run2['summary']['passing_tasks']}/{run2['summary']['total_tasks']}")

# Compare
comparison = benchmark.compare_runs(run1, run2)
print(f"\n=== Comparison ===")
print(f"Run 1: {comparison['avg_overall_a']:.2%}")
print(f"Run 2: {comparison['avg_overall_b']:.2%}")
print(f"Delta: {comparison['delta']:+.2%}")
print(f"Improved: {'YES' if comparison['improved'] else 'NO'}")
print(f"Significant: {'YES' if comparison['significant'] else 'NO'}")

# Per-task breakdown
print(f"\n=== Per-Task Results (Run 2) ===")
for score in run2["scores"]:
    status = "✓" if score["overall"] >= 0.7 else "✗"
    print(f"  {status} {score['task_name'][:40]:40s} {score['overall']:.2%}")
```

---

## Takeaways

1. **Golden datasets are your ground truth** — expert-curated, version-controlled, reviewed
2. **Synthetic datasets scale coverage** — LLM-generated but human-validated
3. **Regression benchmarks** catch quality drops before they reach production
4. **Before/after comparisons** prove that changes are improvements
5. **Version your eval data like code** — it's just as important

## Setting the Stage for Module 14

Phase 3 is complete — you now have a comprehensive evaluation infrastructure. Phase 4 explores **advanced techniques** starting with **fine-tuning**. Sometimes RAG + prompting isn't enough, and you need to teach the model new behaviors. Module 14 covers when and how to fine-tune, using LoRA to make it practical on your M4 Pro.
