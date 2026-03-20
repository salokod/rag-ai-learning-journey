# Module 11: Testing with DeepEval

## Goal
Write `pytest`-style tests for your LLM outputs. When tests pass, you ship with confidence. When they fail, you catch the problem before it reaches the manufacturing floor.

---

## The Idea

What if you could write tests for your LLM the same way you write tests for regular code?

```python
# Regular software test:
def test_add():
    assert add(2, 3) == 5

# LLM test (what we're building):
def test_welding_task_mentions_ppe():
    output = generate_task("TIG welding on aluminum")
    assert "ppe" in output.lower() or "helmet" in output.lower()
```

That is exactly what DeepEval enables. You define what "good" means, and `pytest` enforces it. Every time you change a prompt, retune a model, or update your knowledge base, the tests tell you if anything broke.

Let's build this up piece by piece.

---

## Part 1: Your First LLM Test

### Step 1: Verify installation

```bash
pip install deepeval --quiet
```

Confirm it works:

```bash
python -c "from deepeval.test_case import LLMTestCase; print('DeepEval ready')"
```

### Step 2: Write one test

Create this file:

```python
# 11-testing-with-deepeval/test_first.py
import pytest
import re


def test_task_has_numbered_steps():
    """The LLM output must contain at least 3 numbered steps."""
    # This is the output we're testing
    # (In practice, you'd generate this from your LLM)
    output = """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete on the welding station.
2. Don required PPE: safety glasses, leather gloves, inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6.
4. Measure weld size with fillet gauge -- minimum 6mm leg per SH-4402.
5. Record findings on Form QC-107."""

    steps = re.findall(r'^\s*\d+[\.\)]\s', output, re.MULTILINE)
    assert len(steps) >= 3, f"Expected >= 3 numbered steps, got {len(steps)}"
```

Run it:

```bash
cd 11-testing-with-deepeval
pytest test_first.py -v
```

You should see a green PASSED. That is your first LLM test. Simple, fast, no LLM judge needed.

### Step 3: Make it fail on purpose

Add a second test to the same file:

```python
def test_bad_output_fails():
    """This should FAIL -- proving our test catches bad output."""
    bad_output = "Check the welds."
    steps = re.findall(r'^\s*\d+[\.\)]\s', bad_output, re.MULTILINE)
    assert len(steps) >= 3, f"Expected >= 3 numbered steps, got {len(steps)}"
```

Run again:

```bash
pytest test_first.py -v
```

One passes, one fails. The failure message tells you exactly what went wrong: "Expected >= 3 numbered steps, got 0." That is the point -- tests catch bad output before it reaches anyone.

Now delete or comment out `test_bad_output_fails` (it was just for demonstration). Let's build real tests.

---

## Part 2: Testing a Real Generator

### Step 4: Set up the system under test

```python
# 11-testing-with-deepeval/task_generator.py
"""The task description generator we're testing."""

import ollama


class TaskDescriptionGenerator:
    def __init__(self, model="llama3.1:8b"):
        self.model = model
        self.system_prompt = """You are a manufacturing technical writer.
Write task descriptions with numbered steps (3-8 steps).
Start each step with an action verb.
Include safety requirements and specific references where provided.
Keep under 150 words."""

    def generate(self, task_name, context=""):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Write a task description for: {task_name}"
                + (f"\n\nReference: {context}" if context else ""),
            },
        ]
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.0},
        )
        return response["message"]["content"]
```

Quick sanity check -- run this interactively:

```python
from task_generator import TaskDescriptionGenerator

gen = TaskDescriptionGenerator()
print(gen.generate("Inspect welded joints on steel frame", "AWS D1.1, Form QC-107"))
```

Does it look reasonable? Good. Now let's write tests that enforce "reasonable" automatically.

### Step 5: Format tests

```python
# 11-testing-with-deepeval/test_format.py
"""Test that generated task descriptions meet structural requirements."""

import pytest
import re
from task_generator import TaskDescriptionGenerator

generator = TaskDescriptionGenerator()


@pytest.fixture(scope="module")
def weld_task():
    """Generate once, reuse across tests (LLM calls are slow)."""
    return generator.generate(
        "Inspect welded joints on steel frame",
        context="AWS D1.1, Form QC-107 for reporting",
    )


@pytest.fixture(scope="module")
def cnc_task():
    return generator.generate(
        "Set up CNC lathe for shaft machining",
        context="Drawing SH-4402, tolerance +/-0.005 inch",
    )


def test_has_numbered_steps(weld_task):
    steps = re.findall(r'^\s*\d+[\.\)]', weld_task, re.MULTILINE)
    assert len(steps) >= 3, f"Only {len(steps)} numbered steps found"


def test_word_count(weld_task):
    words = len(weld_task.split())
    assert 30 <= words <= 200, f"Word count {words} outside 30-200 range"


def test_uses_action_verbs(weld_task):
    verbs = [
        "inspect", "verify", "check", "ensure", "record",
        "apply", "install", "remove", "measure", "don",
        "confirm", "clean", "document", "perform",
    ]
    text_lower = weld_task.lower()
    found = [v for v in verbs if v in text_lower]
    assert len(found) >= 2, f"Only found verbs: {found}"


def test_references_provided_spec(weld_task):
    has_ref = "AWS" in weld_task or "D1.1" in weld_task or "QC-107" in weld_task
    assert has_ref, "Missing reference to provided specifications (AWS D1.1 or QC-107)"


def test_cnc_also_has_steps(cnc_task):
    steps = re.findall(r'^\s*\d+[\.\)]', cnc_task, re.MULTILINE)
    assert len(steps) >= 3, f"CNC task only has {len(steps)} steps"
```

Run it:

```bash
pytest test_format.py -v
```

Five tests, all checking structure. These run in seconds (one LLM call each for the two fixtures, then pure regex). If any future prompt change breaks the format, you will know immediately.

---

## Part 3: Safety-Critical Tests

This is where testing earns its keep. In manufacturing, missing a safety callout is not a minor bug -- it is a liability.

### Step 6: Welding tasks MUST mention PPE

```python
# 11-testing-with-deepeval/test_safety.py
"""Safety-critical tests. These MUST pass. No exceptions."""

import pytest
from task_generator import TaskDescriptionGenerator

generator = TaskDescriptionGenerator()


class TestSafetyCritical:

    @pytest.mark.parametrize("task,safety_terms", [
        (
            "Perform TIG welding on aluminum frame",
            ["helmet", "gloves", "ppe", "protective", "ventilation", "fume"],
        ),
        (
            "Operate hydraulic press for stamping",
            ["lockout", "tagout", "guard", "safety", "two-hand", "interlock"],
        ),
        (
            "Use angle grinder to deburr steel parts",
            ["glasses", "shield", "guard", "ppe", "gloves", "safety"],
        ),
        (
            "Load parts into industrial oven for heat treatment",
            ["gloves", "burns", "temperature", "ppe", "safety", "heat"],
        ),
    ])
    def test_high_risk_task_includes_safety(self, task, safety_terms):
        """High-risk tasks MUST mention at least one relevant safety term."""
        output = generator.generate(task).lower()
        matches = [t for t in safety_terms if t in output]
        assert len(matches) >= 1, (
            f"SAFETY FAILURE: '{task}' output mentions NONE of {safety_terms}"
        )

    @pytest.mark.parametrize("task", [
        "Perform MIG welding on carbon steel",
        "Operate CNC mill with coolant system",
        "Use plasma cutter on sheet metal",
    ])
    def test_high_risk_never_omits_ppe(self, task):
        """PPE-required tasks must say 'PPE' or list specific equipment."""
        output = generator.generate(task).lower()
        ppe_indicators = [
            "ppe", "personal protective", "safety glasses", "helmet",
            "gloves", "face shield", "ear protection", "steel-toe",
        ]
        has_ppe = any(term in output for term in ppe_indicators)
        assert has_ppe, f"PPE not mentioned for high-risk task: {task}"
```

Run it:

```bash
pytest test_safety.py -v
```

If any of these fail, your prompt needs work before the system goes to production. These are your non-negotiable gates.

---

## Part 4: Consistency Tests

### Step 7: Same input, same output

At temperature 0, the same prompt should give the same answer every time. If it doesn't, you have a reproducibility problem:

```python
# 11-testing-with-deepeval/test_consistency.py
"""Test output consistency and reproducibility."""

import pytest
from task_generator import TaskDescriptionGenerator

generator = TaskDescriptionGenerator()


def test_deterministic_at_temp_zero():
    """Same input at temperature 0 should produce identical output."""
    task = "Calibrate digital pressure gauge"
    output1 = generator.generate(task)
    output2 = generator.generate(task)
    assert output1 == output2, (
        "Outputs differ at temperature 0. "
        "This means results are not reproducible.\n"
        f"Output 1: {output1[:100]}...\n"
        f"Output 2: {output2[:100]}..."
    )


def test_format_consistent_across_tasks():
    """Different tasks should all follow the same structural format."""
    tasks = [
        "Inspect incoming raw material",
        "Perform daily forklift safety check",
        "Replace worn conveyor belt rollers",
    ]
    import re

    for task in tasks:
        output = generator.generate(task)
        steps = re.findall(r'^\s*\d+[\.\)]', output, re.MULTILINE)
        words = len(output.split())
        assert len(steps) >= 3, f"'{task}' has only {len(steps)} steps"
        assert words <= 250, f"'{task}' too long at {words} words"
```

Run it:

```bash
pytest test_consistency.py -v
```

If the determinism test fails, investigate: is your Ollama version different? Is there GPU non-determinism? This test catches subtle infrastructure issues.

---

## Part 5: Custom Metrics with DeepEval

DeepEval's built-in metrics (AnswerRelevancyMetric, FaithfulnessMetric, etc.) use OpenAI by default. Since we are running locally, let's build custom metrics that are specific to manufacturing and work without any API keys.

### Step 8: A manufacturing format metric

```python
# 11-testing-with-deepeval/custom_metrics.py
"""Custom DeepEval metrics for manufacturing task descriptions."""

import re
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ManufacturingFormatMetric(BaseMetric):
    """Does the task description follow manufacturing format standards?"""

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.score = 0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Manufacturing Format Compliance"

    def measure(self, test_case: LLMTestCase) -> float:
        text = test_case.actual_output
        checks = {
            "has_numbered_steps": len(
                re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)
            ) >= 3,
            "reasonable_length": 30 <= len(text.split()) <= 200,
            "has_action_verbs": any(
                v in text.lower()
                for v in [
                    "inspect", "verify", "check", "install",
                    "remove", "record", "ensure", "measure",
                ]
            ),
            "has_references": bool(re.search(r'[A-Z]{2,}-\d{2,}', text)),
            "mostly_active_voice": sum(
                1 for p in ["should be", "is to be", "are to be"]
                if p in text.lower()
            ) <= 1,
        }

        self.score = sum(checks.values()) / len(checks)
        failed = [k for k, v in checks.items() if not v]
        passed_count = sum(checks.values())
        self.reason = (
            f"Passed {passed_count}/{len(checks)} checks."
            + (f" Failed: {failed}" if failed else " All passed.")
        )
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self):
        return self.success

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


class SafetyInclusionMetric(BaseMetric):
    """Are safety requirements included for hazardous tasks?"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.score = 0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Safety Inclusion"

    def measure(self, test_case: LLMTestCase) -> float:
        text = test_case.actual_output.lower()
        task = test_case.input.lower()

        high_risk = [
            "weld", "press", "grind", "cut", "drill",
            "lathe", "mill", "forklift", "crane", "chemical",
            "oven", "furnace", "plasma", "solder",
        ]
        is_high_risk = any(kw in task for kw in high_risk)

        safety_terms = [
            "ppe", "safety", "glasses", "gloves", "helmet",
            "protection", "lockout", "tagout", "guard",
            "caution", "warning", "hazard", "shield",
        ]
        mentions = sum(1 for t in safety_terms if t in text)

        if is_high_risk:
            self.score = min(mentions / 3, 1.0)
            self.reason = f"High-risk task. {mentions} safety terms found (need 3+)."
        else:
            self.score = 1.0 if mentions > 0 else 0.5
            self.reason = f"Standard task. {mentions} safety terms found."

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self):
        return self.success

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)
```

### Step 9: Use custom metrics in tests

```python
# 11-testing-with-deepeval/test_custom_metrics.py
"""Tests using our custom manufacturing metrics."""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from task_generator import TaskDescriptionGenerator
from custom_metrics import ManufacturingFormatMetric, SafetyInclusionMetric

generator = TaskDescriptionGenerator()


@pytest.mark.parametrize("task,context", [
    (
        "Inspect weld joints on frame assembly",
        "AWS D1.1, Form QC-107",
    ),
    (
        "Calibrate digital pressure gauge",
        "NIST-traceable standard, Form CAL-201",
    ),
    (
        "Replace worn conveyor belt rollers",
        "P/N CR-200 series, Form PM-105",
    ),
])
def test_manufacturing_format(task, context):
    output = generator.generate(task, context)
    test_case = LLMTestCase(input=task, actual_output=output)
    format_metric = ManufacturingFormatMetric(threshold=0.6)
    assert_test(test_case, [format_metric])


@pytest.mark.parametrize("task", [
    "Perform MIG welding on carbon steel frame",
    "Operate hydraulic press for bracket stamping",
    "Use angle grinder to deburr steel castings",
])
def test_safety_inclusion(task):
    output = generator.generate(task)
    test_case = LLMTestCase(input=task, actual_output=output)
    safety_metric = SafetyInclusionMetric(threshold=0.5)
    assert_test(test_case, [safety_metric])
```

Run it:

```bash
pytest test_custom_metrics.py -v
```

Each test generates a task description, runs it through your custom metric, and passes or fails based on the threshold. You can see which specific checks failed in the output.

---

## Part 6: The Full Test Suite

### Step 10: One file that runs everything

Let's consolidate into a comprehensive test suite:

```python
# 11-testing-with-deepeval/test_suite.py
"""
Complete test suite for manufacturing task description quality.

Run with:  pytest test_suite.py -v
"""

import pytest
import re
from task_generator import TaskDescriptionGenerator
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from custom_metrics import ManufacturingFormatMetric, SafetyInclusionMetric

generator = TaskDescriptionGenerator()


# =========================================================
# STRUCTURE TESTS -- fast, no LLM judge
# =========================================================

class TestStructure:
    """Every task description must meet basic structural requirements."""

    @pytest.fixture(scope="class")
    def sample_output(self):
        return generator.generate(
            "Inspect welded joints on steel frame",
            context="AWS D1.1, Form QC-107",
        )

    def test_has_numbered_steps(self, sample_output):
        steps = re.findall(r'^\s*\d+[\.\)]', sample_output, re.MULTILINE)
        assert len(steps) >= 3

    def test_word_count_in_range(self, sample_output):
        words = len(sample_output.split())
        assert 30 <= words <= 200

    def test_starts_with_action_verbs(self, sample_output):
        verbs = [
            "inspect", "verify", "check", "ensure", "record",
            "measure", "don", "apply", "install", "remove",
        ]
        found = [v for v in verbs if v in sample_output.lower()]
        assert len(found) >= 2

    def test_not_excessively_verbose(self, sample_output):
        # Guard against wordy, bureaucratic output
        sentences = sample_output.split(".")
        long_sentences = [s for s in sentences if len(s.split()) > 30]
        assert len(long_sentences) <= 2, "Too many overly long sentences"


# =========================================================
# SAFETY TESTS -- non-negotiable for manufacturing
# =========================================================

class TestSafety:
    """Safety content must be present for hazardous tasks."""

    @pytest.mark.parametrize("task,must_mention_one_of", [
        (
            "Perform TIG welding on aluminum",
            ["helmet", "gloves", "ppe", "protective", "shield"],
        ),
        (
            "Operate hydraulic press",
            ["lockout", "tagout", "guard", "safety", "loto"],
        ),
        (
            "Use angle grinder on steel",
            ["glasses", "shield", "ppe", "guard", "safety"],
        ),
    ])
    def test_hazardous_task_safety(self, task, must_mention_one_of):
        output = generator.generate(task).lower()
        matches = [t for t in must_mention_one_of if t in output]
        assert matches, (
            f"SAFETY GAP: '{task}' mentions none of {must_mention_one_of}"
        )


# =========================================================
# CONSISTENCY TESTS -- reproducibility matters
# =========================================================

class TestConsistency:

    def test_same_input_same_output(self):
        task = "Calibrate digital pressure gauge"
        a = generator.generate(task)
        b = generator.generate(task)
        assert a == b, "Non-deterministic output at temperature 0"

    def test_all_tasks_have_steps(self):
        tasks = [
            "Inspect incoming raw material",
            "Perform daily forklift check",
            "Replace conveyor belt rollers",
        ]
        for task in tasks:
            output = generator.generate(task)
            steps = re.findall(r'^\s*\d+[\.\)]', output, re.MULTILINE)
            assert len(steps) >= 3, f"'{task}' only has {len(steps)} steps"


# =========================================================
# CUSTOM METRIC TESTS -- using DeepEval framework
# =========================================================

class TestCustomMetrics:

    @pytest.mark.parametrize("task,context", [
        ("Inspect weld joints", "AWS D1.1, QC-107"),
        ("Calibrate pressure gauge", "Form CAL-201"),
    ])
    def test_format_metric(self, task, context):
        output = generator.generate(task, context)
        test_case = LLMTestCase(input=task, actual_output=output)
        assert_test(test_case, [ManufacturingFormatMetric(threshold=0.6)])

    @pytest.mark.parametrize("task", [
        "Perform MIG welding on steel",
        "Operate hydraulic press",
    ])
    def test_safety_metric(self, task):
        output = generator.generate(task)
        test_case = LLMTestCase(input=task, actual_output=output)
        assert_test(test_case, [SafetyInclusionMetric(threshold=0.5)])
```

Run the whole suite:

```bash
pytest test_suite.py -v --tb=short
```

You should see a clear report: how many passed, how many failed, and exactly what went wrong for each failure.

---

## Part 7: CI/CD Integration

### Step 11: How this fits in your workflow

Here is the picture. Every time you change a prompt, update your knowledge base, or upgrade a model:

```
You make a change
       |
       v
  git commit
       |
       v
  CI runs: pytest test_suite.py
       |
       +--> All pass? --> Ship it
       |
       +--> Something fails? --> Fix before shipping
```

To run DeepEval tests in CI, your pipeline just needs:

```yaml
# .github/workflows/llm-tests.yml (example)
- name: Run LLM tests
  run: |
    pip install -r requirements.txt
    pytest 11-testing-with-deepeval/test_suite.py -v --tb=short
```

The key insight: **your LLM tests run exactly like any other test suite.** No special infrastructure. Your team already knows pytest. The only difference is that these tests call an LLM, so they take seconds instead of milliseconds.

### Step 12: Practical tips for LLM testing

A few things you will learn the hard way (so let's save you the time):

**Cache LLM outputs during development.** LLM calls are slow. Use `scope="module"` or `scope="class"` on pytest fixtures so you generate once and test many things on the same output.

**Separate fast tests from slow tests.** Use pytest markers:

```python
@pytest.mark.slow
def test_that_needs_llm_call():
    ...

@pytest.mark.fast
def test_pure_heuristic():
    ...
```

Then run fast tests often, slow tests before commits:

```bash
pytest -m fast           # Quick feedback loop
pytest -m "not slow"     # Same thing
pytest                   # Everything before shipping
```

**Keep thresholds practical.** Start with lenient thresholds (0.5-0.6) and tighten as your system improves. Starting too strict means everything fails and the tests become noise.

**Test the things that matter.** For manufacturing: safety content, numbered steps, specific references. Don't test for exact wording -- test for structural and safety properties.

---

## Takeaways

1. **LLM tests work exactly like regular pytest tests** -- your team already knows how to use them.
2. **Heuristic tests are free and fast** -- use them for format, structure, and safety checks on every output.
3. **Custom metrics encode YOUR quality standards** -- not generic ones from a framework.
4. **Parametrized tests catch class-wide failures** -- all welding tasks need PPE, not just the one you happened to test.
5. **Same input at temp=0 should give same output** -- if it doesn't, investigate.
6. **CI/CD integration is just `pytest`** -- nothing special needed.

## What's Next

You can now test and evaluate. But in production, you need **observability** -- seeing what is happening in real time, tracking performance over time, debugging failures, and managing prompt versions. Module 12 introduces **Langfuse**, an open-source observability platform that gives you a production-grade dashboard for your LLM system.
