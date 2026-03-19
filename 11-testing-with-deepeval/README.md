# Module 11: Testing with DeepEval

## Goal
Write pytest-style tests for your LLM outputs using DeepEval. Build a test suite that catches regressions and runs in CI/CD, just like traditional software tests.

---

## Concepts

### DeepEval: pytest for LLMs

DeepEval treats LLM evaluation like software testing:
- Write test cases with expected behavior
- Run with `pytest` or `deepeval test run`
- Get pass/fail results with detailed metrics
- Integrate into CI/CD pipelines

### DeepEval vs. Ragas

| | DeepEval | Ragas |
|---|---|---|
| **Focus** | General LLM testing | RAG-specific evaluation |
| **Interface** | pytest-compatible | Dataset-based evaluation |
| **Metrics** | 14+ metrics for all LLM tasks | RAG-focused metrics |
| **Best for** | CI/CD, regression testing | RAG pipeline evaluation |
| **Use together?** | Yes! Different strengths | Yes! |

---

## Exercise 1: Your First DeepEval Test

```python
# 11-testing-with-deepeval/ex1_first_test.py
"""Write your first LLM test with DeepEval."""

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
import ollama

# Generate a response to test
question = "What PPE is required for MIG welding per company standards?"
context = [
    "PPE for welding: Auto-darkening helmet (shade 10-13), leather welding gloves, "
    "FR clothing, steel-toe boots. Safety glasses must be worn under the helmet."
]

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Context: {context[0]}\n\nQuestion: {question}"},
    ],
    options={"temperature": 0.0},
)

actual_output = response["message"]["content"]

# Create a test case
test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    retrieval_context=context,
)

# Define metrics
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,  # Must score above 0.7 to pass
    model="gpt-4o",  # DeepEval uses this for judging — see note below
)

# NOTE: DeepEval's built-in metrics use OpenAI by default.
# For fully local evaluation, you can:
# 1. Set OPENAI_API_BASE=http://localhost:11434/v1 and OPENAI_API_KEY=ollama
# 2. Or use custom metrics (Exercise 3)

print("=== Test Case ===")
print(f"Question: {question}")
print(f"Context: {context[0][:80]}...")
print(f"Answer: {actual_output[:200]}...")

# For local testing without OpenAI, use custom metrics (see Exercise 3)
# To run with OpenAI: assert_test(test_case, [relevancy_metric])
print("\n=== DeepEval Test Structure ===")
print("1. LLMTestCase: input + actual_output + context")
print("2. Metrics: what to measure (relevancy, faithfulness, etc.)")
print("3. assert_test: pass if all metrics above threshold")
print("4. Run with: deepeval test run <file> or pytest <file>")
```

---

## Exercise 2: Building a Test Suite

```python
# 11-testing-with-deepeval/test_task_descriptions.py
"""A complete test suite for manufacturing task description quality.
Run with: pytest 11-testing-with-deepeval/test_task_descriptions.py -v"""

import pytest
import ollama
import json
import re


class TaskDescriptionGenerator:
    """The system under test — our task description generator."""

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.system_prompt = """You are a manufacturing technical writer.
Write task descriptions with numbered steps (3-5 steps).
Start each step with an action verb.
Include safety requirements and specific references.
Keep under 150 words."""

    def generate(self, task_name: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Write a task description for: {task_name}\n\nReference: {context}"},
        ]
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.0},
        )
        return response["message"]["content"]


# Initialize the generator
generator = TaskDescriptionGenerator()


# === HEURISTIC TESTS (fast, no LLM judge needed) ===

class TestTaskDescriptionFormat:
    """Test that task descriptions meet structural requirements."""

    @pytest.fixture
    def weld_inspection_task(self):
        return generator.generate(
            "Inspect welded joints on steel frame",
            context="AWS D1.1, Form QC-107 for reporting",
        )

    @pytest.fixture
    def cnc_setup_task(self):
        return generator.generate(
            "Set up CNC lathe for shaft machining",
            context="Drawing SH-4402, tolerance ±0.005 inch",
        )

    def test_has_numbered_steps(self, weld_inspection_task):
        """Task description must have at least 3 numbered steps."""
        steps = re.findall(r'^\s*\d+[\.\)]', weld_inspection_task, re.MULTILINE)
        assert len(steps) >= 3, f"Expected ≥3 numbered steps, found {len(steps)}"

    def test_word_count_reasonable(self, weld_inspection_task):
        """Task description should be 30-200 words."""
        words = len(weld_inspection_task.split())
        assert 30 <= words <= 200, f"Word count {words} outside 30-200 range"

    def test_uses_action_verbs(self, weld_inspection_task):
        """Steps should start with action verbs."""
        action_verbs = ["inspect", "verify", "check", "ensure", "record",
                        "apply", "install", "remove", "measure", "don",
                        "confirm", "clean", "document", "perform"]
        text_lower = weld_inspection_task.lower()
        verb_count = sum(1 for v in action_verbs if v in text_lower)
        assert verb_count >= 2, f"Found only {verb_count} action verbs"

    def test_mentions_safety(self, weld_inspection_task):
        """Welding task must mention safety/PPE."""
        safety_terms = ["ppe", "safety", "helmet", "gloves", "glasses",
                        "protection", "protective", "lockout", "caution"]
        text_lower = weld_inspection_task.lower()
        has_safety = any(term in text_lower for term in safety_terms)
        assert has_safety, "No safety/PPE mentions in welding task description"

    def test_references_specification(self, weld_inspection_task):
        """Should reference the provided specification."""
        # Should mention AWS D1.1 or QC-107 since we provided them as context
        has_spec_ref = "AWS" in weld_inspection_task or "QC-107" in weld_inspection_task or "D1.1" in weld_inspection_task
        assert has_spec_ref, "Missing reference to provided specifications"

    def test_cnc_has_numbered_steps(self, cnc_setup_task):
        """CNC setup task must also have numbered steps."""
        steps = re.findall(r'^\s*\d+[\.\)]', cnc_setup_task, re.MULTILINE)
        assert len(steps) >= 3


# === CONSISTENCY TESTS ===

class TestTaskDescriptionConsistency:
    """Test that the same input produces consistent output."""

    def test_deterministic_output(self):
        """At temperature 0, same input should give same output."""
        task = "Calibrate digital pressure gauge"
        output1 = generator.generate(task)
        output2 = generator.generate(task)
        # They should be identical or very similar at temp=0
        assert output1 == output2, "Outputs differ at temperature 0 — non-deterministic"

    def test_format_consistency_across_tasks(self):
        """Different tasks should all follow the same format."""
        tasks = [
            "Inspect incoming raw material",
            "Perform daily forklift check",
            "Replace worn conveyor rollers",
        ]

        for task in tasks:
            output = generator.generate(task)
            steps = re.findall(r'^\s*\d+[\.\)]', output, re.MULTILINE)
            words = len(output.split())
            assert len(steps) >= 3, f"Task '{task}' has only {len(steps)} steps"
            assert words <= 250, f"Task '{task}' too long: {words} words"


# === SAFETY-CRITICAL TESTS ===

class TestSafetyCriticalContent:
    """Ensure safety-critical content is never omitted."""

    @pytest.mark.parametrize("task,expected_safety_content", [
        ("Operate hydraulic press", ["lockout", "tagout", "two-hand", "guard", "safety", "interlock"]),
        ("Perform TIG welding on aluminum", ["helmet", "gloves", "ventilation", "ppe", "fume", "protective"]),
        ("Use angle grinder to deburr parts", ["glasses", "shield", "guard", "ppe", "gloves", "safety"]),
    ])
    def test_high_risk_tasks_mention_safety(self, task, expected_safety_content):
        """High-risk tasks MUST include relevant safety content."""
        output = generator.generate(task).lower()
        matches = [term for term in expected_safety_content if term in output]
        assert len(matches) >= 1, (
            f"Safety-critical task '{task}' missing safety terms. "
            f"Expected at least one of {expected_safety_content}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

---

## Exercise 3: Custom Metrics for Manufacturing

```python
# 11-testing-with-deepeval/ex3_custom_metrics.py
"""Build custom DeepEval metrics specific to manufacturing task descriptions."""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import ollama
import json
import re


class ManufacturingFormatMetric(BaseMetric):
    """Custom metric: Does the task description follow manufacturing format?"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0
        self.reason = ""

    @property
    def __name__(self):
        return "Manufacturing Format Compliance"

    def measure(self, test_case: LLMTestCase) -> float:
        text = test_case.actual_output
        checks = {
            "has_numbered_steps": bool(re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)),
            "reasonable_length": 30 <= len(text.split()) <= 200,
            "has_action_verbs": any(v in text.lower() for v in
                ["inspect", "verify", "check", "install", "remove", "record", "ensure"]),
            "has_references": bool(re.search(r'[A-Z]{2,}-\d{2,}', text)),
            "active_voice": sum(1 for p in ["should be", "is to be"]
                if p in text.lower()) <= 1,
        }

        self.score = sum(checks.values()) / len(checks)
        failed = [k for k, v in checks.items() if not v]
        self.reason = f"Passed {sum(checks.values())}/{len(checks)}. " + \
                      (f"Failed: {failed}" if failed else "All checks passed.")
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


class SafetyInclusionMetric(BaseMetric):
    """Custom metric: Are safety requirements included where needed?"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.score = 0
        self.reason = ""

    @property
    def __name__(self):
        return "Safety Inclusion"

    def measure(self, test_case: LLMTestCase) -> float:
        text = test_case.actual_output.lower()
        task = test_case.input.lower()

        # Determine if this is a safety-critical task
        high_risk_keywords = ["weld", "press", "grind", "cut", "drill",
                              "lathe", "mill", "forklift", "crane", "chemical"]
        is_high_risk = any(kw in task for kw in high_risk_keywords)

        safety_terms = ["ppe", "safety", "glasses", "gloves", "helmet",
                        "protection", "lockout", "tagout", "guard", "caution",
                        "warning", "hazard"]
        safety_mentions = sum(1 for t in safety_terms if t in text)

        if is_high_risk:
            # High-risk tasks need strong safety content
            self.score = min(safety_mentions / 3, 1.0)  # Need at least 3 safety terms
            self.reason = f"High-risk task with {safety_mentions} safety mentions"
        else:
            # Low-risk tasks get a pass with any mention
            self.score = min(safety_mentions / 1, 1.0) if safety_mentions > 0 else 0.5
            self.reason = f"Low-risk task with {safety_mentions} safety mentions"

        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


# Test with our custom metrics
format_metric = ManufacturingFormatMetric(threshold=0.7)
safety_metric = SafetyInclusionMetric(threshold=0.5)

# Generate test outputs
tasks = [
    ("Inspect weld joints on frame assembly", "AWS D1.1, Form QC-107"),
    ("File purchase order for office supplies", ""),
    ("Operate CNC mill for aluminum housing", "Drawing HG-2200-Rev.C"),
]

for task_name, context in tasks:
    output = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "Write a manufacturing task description with numbered steps. Include safety and references."},
            {"role": "user", "content": f"Task: {task_name}\nContext: {context}"},
        ],
        options={"temperature": 0.0},
    )["message"]["content"]

    test_case = LLMTestCase(input=task_name, actual_output=output)

    format_metric.measure(test_case)
    safety_metric.measure(test_case)

    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"Format: {format_metric.score:.2f} ({'PASS' if format_metric.is_successful() else 'FAIL'}) — {format_metric.reason}")
    print(f"Safety: {safety_metric.score:.2f} ({'PASS' if safety_metric.is_successful() else 'FAIL'}) — {safety_metric.reason}")
```

---

## Takeaways

1. **DeepEval lets you write LLM tests like software tests** — pytest-compatible, CI/CD-ready
2. **Heuristic tests are free and fast** — use them for format, structure, and safety checks
3. **Custom metrics** let you encode YOUR company's quality standards
4. **Parametrized tests** catch class-wide failures (all welding tasks need safety content)
5. **Run tests before AND after changes** — this is your regression safety net

## Setting the Stage for Module 12

You can test and evaluate. But in production, you need **observability** — seeing what's happening in real time, tracking costs, debugging failures, and managing prompt versions. Module 12 introduces **Langfuse**, the open-source Galileo alternative that gives you a production-grade observability dashboard.
