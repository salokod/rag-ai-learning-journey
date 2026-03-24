# Module 11: Testing with DeepEval

## Goal
Write `pytest`-style tests for your LLM outputs. When tests pass, you ship with confidence. When they fail, you catch the problem before it reaches the draft board.

---

## The Idea

What if you could write tests for your LLM the same way you write tests for regular code?

```python
# Regular software test:
def test_add():
    assert add(2, 3) == 5

# LLM test (what we're building):
def test_qb_report_mentions_arm_strength():
    output = generate_report("QB prospect, pocket passer")
    assert "arm strength" in output.lower() or "release" in output.lower()
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


def test_report_has_numbered_steps():
    """The LLM output must contain at least 3 numbered evaluation points."""
    # This is the output we're testing
    # (In practice, you'd generate this from your LLM)
    output = """SCOUTING REPORT: QB PROSPECT – MARCUS ALLEN

1. Pocket passer with elite accuracy, completing 68% of passes with a 2.3-second average release.
2. Excels on intermediate routes (15-25 yards), consistently hitting tight windows.
3. Reads defenses pre-snap and adjusts protection calls at the line.
4. Arm strength measured at 62 mph at the combine.
5. Weakness: locks onto first read under pressure, leading to forced throws."""

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
    bad_output = "He looks like a good quarterback."
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
# 11-testing-with-deepeval/report_generator.py
"""The scouting report generator we're testing."""

from openai import OpenAI


class ScoutingReportGenerator:
    def __init__(self, model="gemma3:12b"):
        self.model = model
        self.llm = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        self.system_prompt = """You are an NFL draft analyst preparing scouting reports.
Write scouting reports with numbered evaluation points (3-8 points).
Start each point with a descriptive assessment.
Include measurables, stats, and specific weaknesses where provided.
Keep under 150 words."""

    def generate(self, player_description, context=""):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Write a scouting report for: {player_description}"
                + (f"\n\nReference: {context}" if context else ""),
            },
        ]
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content
```

Quick sanity check -- run this interactively:

```python
from report_generator import ScoutingReportGenerator

gen = ScoutingReportGenerator()
print(gen.generate("QB prospect, pocket passer with elite accuracy", "QB-101, 68% completion, 62 mph arm strength"))
```

Does it look reasonable? Good. Now let's write tests that enforce "reasonable" automatically.

### Step 5: Format tests

```python
# 11-testing-with-deepeval/test_format.py
"""Test that generated scouting reports meet structural requirements."""

import pytest
import re
from report_generator import ScoutingReportGenerator

generator = ScoutingReportGenerator()


@pytest.fixture(scope="module")
def qb_report():
    """Generate once, reuse across tests (LLM calls are slow)."""
    return generator.generate(
        "QB prospect, pocket passer with elite accuracy",
        context="QB-101, 68% completion rate, 62 mph arm strength",
    )


@pytest.fixture(scope="module")
def rb_report():
    return generator.generate(
        "RB prospect, explosive runner with receiving ability",
        context="RB-201, 4.38 40-yard dash, 3.8 yards after contact",
    )


def test_has_numbered_steps(qb_report):
    steps = re.findall(r'^\s*\d+[\.\)]', qb_report, re.MULTILINE)
    assert len(steps) >= 3, f"Only {len(steps)} numbered steps found"


def test_word_count(qb_report):
    words = len(qb_report.split())
    assert 30 <= words <= 200, f"Word count {words} outside 30-200 range"


def test_uses_scouting_language(qb_report):
    terms = [
        "accuracy", "arm", "release", "completion", "reads",
        "pocket", "routes", "strength", "weakness", "projection",
        "draft", "prospect", "grade", "elite", "measurables",
    ]
    text_lower = qb_report.lower()
    found = [t for t in terms if t in text_lower]
    assert len(found) >= 2, f"Only found scouting terms: {found}"


def test_references_provided_stats(qb_report):
    has_ref = "68%" in qb_report or "62 mph" in qb_report or "QB-101" in qb_report
    assert has_ref, "Missing reference to provided stats (68%, 62 mph, or QB-101)"


def test_rb_also_has_steps(rb_report):
    steps = re.findall(r'^\s*\d+[\.\)]', rb_report, re.MULTILINE)
    assert len(steps) >= 3, f"RB report only has {len(steps)} steps"
```

Run it:

```bash
pytest test_format.py -v
```

Five tests, all checking structure. These run in seconds (one LLM call each for the two fixtures, then pure regex). If any future prompt change breaks the format, you will know immediately.

---

## Part 3: Critical Content Tests

This is where testing earns its keep. In football scouting, missing key weaknesses or fabricating stats is not a minor bug -- it can cost a franchise a draft pick worth millions.

### Step 6: Scouting reports MUST mention weaknesses

```python
# 11-testing-with-deepeval/test_completeness.py
"""Completeness tests. These MUST pass. No exceptions."""

import pytest
from report_generator import ScoutingReportGenerator

generator = ScoutingReportGenerator()


class TestCompleteness:

    @pytest.mark.parametrize("player,weakness_terms", [
        (
            "QB prospect, pocket passer under pressure",
            ["weakness", "concern", "limitation", "struggles", "needs work", "inconsistent"],
        ),
        (
            "RB prospect, explosive runner with receiving ability",
            ["weakness", "concern", "limitation", "pass protection", "blitz", "blocking"],
        ),
        (
            "WR prospect, route runner with speed",
            ["weakness", "concern", "press", "physical", "drop", "limitation"],
        ),
        (
            "OL prospect, pass protection specialist",
            ["weakness", "concern", "combo", "run blocking", "mobility", "limitation"],
        ),
    ])
    def test_report_includes_weaknesses(self, player, weakness_terms):
        """Every scouting report MUST identify at least one weakness."""
        output = generator.generate(player).lower()
        matches = [t for t in weakness_terms if t in output]
        assert len(matches) >= 1, (
            f"COMPLETENESS FAILURE: '{player}' report mentions NONE of {weakness_terms}"
        )

    @pytest.mark.parametrize("player", [
        "QB prospect with elite arm strength",
        "RB prospect with 4.3 speed",
        "WR prospect with elite route running",
    ])
    def test_report_includes_measurables(self, player):
        """Reports for athletic positions must reference measurables or stats."""
        output = generator.generate(player).lower()
        measurable_indicators = [
            "speed", "40-yard", "vertical", "arm strength", "mph",
            "yards", "completion", "release", "bench", "shuttle",
        ]
        has_measurables = any(term in output for term in measurable_indicators)
        assert has_measurables, f"Measurables not mentioned for: {player}"
```

Run it:

```bash
pytest test_completeness.py -v
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
from report_generator import ScoutingReportGenerator

generator = ScoutingReportGenerator()


def test_deterministic_at_temp_zero():
    """Same input at temperature 0 should produce identical output."""
    player = "QB prospect, pocket passer with elite accuracy"
    output1 = generator.generate(player)
    output2 = generator.generate(player)
    assert output1 == output2, (
        "Outputs differ at temperature 0. "
        "This means results are not reproducible.\n"
        f"Output 1: {output1[:100]}...\n"
        f"Output 2: {output2[:100]}..."
    )


def test_format_consistent_across_positions():
    """Different position reports should all follow the same structural format."""
    players = [
        "QB prospect, strong-armed passer",
        "RB prospect, explosive runner",
        "WR prospect, elite route runner",
    ]
    import re

    for player in players:
        output = generator.generate(player)
        steps = re.findall(r'^\s*\d+[\.\)]', output, re.MULTILINE)
        words = len(output.split())
        assert len(steps) >= 3, f"'{player}' has only {len(steps)} steps"
        assert words <= 250, f"'{player}' too long at {words} words"
```

Run it:

```bash
pytest test_consistency.py -v
```

If the determinism test fails, investigate: is your Ollama version different? Is there GPU non-determinism? This test catches subtle infrastructure issues.

---

## Part 5: Custom Metrics with DeepEval

DeepEval's built-in metrics (AnswerRelevancyMetric, FaithfulnessMetric, etc.) use OpenAI by default. Since we are running locally, let's build custom metrics that are specific to football scouting and work without any API keys.

### Step 8: A scouting report format metric

```python
# 11-testing-with-deepeval/custom_metrics.py
"""Custom DeepEval metrics for NFL scouting reports."""

import re
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ScoutingReportFormatMetric(BaseMetric):
    """Does the scouting report follow proper format standards?"""

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.score = 0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Scouting Report Format Compliance"

    def measure(self, test_case: LLMTestCase) -> float:
        text = test_case.actual_output
        checks = {
            "has_numbered_steps": len(
                re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)
            ) >= 3,
            "reasonable_length": 30 <= len(text.split()) <= 200,
            "has_scouting_language": any(
                v in text.lower()
                for v in [
                    "strength", "weakness", "projection", "prospect",
                    "draft", "measurables", "routes", "coverage",
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


class WeaknessInclusionMetric(BaseMetric):
    """Are weaknesses and concerns identified in the scouting report?"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.score = 0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Weakness Inclusion"

    def measure(self, test_case: LLMTestCase) -> float:
        text = test_case.actual_output.lower()
        player = test_case.input.lower()

        high_value = [
            "quarterback", "qb", "running back", "rb", "wide receiver", "wr",
            "offensive line", "ol", "cornerback", "cb", "edge", "defensive",
        ]
        is_high_value = any(kw in player for kw in high_value)

        weakness_terms = [
            "weakness", "concern", "limitation", "struggles",
            "needs work", "inconsistent", "liability", "risk",
            "injury", "durability", "lacks", "limited",
        ]
        mentions = sum(1 for t in weakness_terms if t in text)

        if is_high_value:
            self.score = min(mentions / 2, 1.0)
            self.reason = f"High-value position. {mentions} weakness indicators found (need 2+)."
        else:
            self.score = 1.0 if mentions > 0 else 0.5
            self.reason = f"Standard evaluation. {mentions} weakness indicators found."

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
"""Tests using our custom scouting report metrics."""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from report_generator import ScoutingReportGenerator
from custom_metrics import ScoutingReportFormatMetric, WeaknessInclusionMetric

generator = ScoutingReportGenerator()


@pytest.mark.parametrize("player,context", [
    (
        "QB prospect, pocket passer with elite accuracy",
        "QB-101, 68% completion rate",
    ),
    (
        "RB prospect, explosive runner with receiving skills",
        "RB-201, 4.38 40-yard dash, 3.8 yards after contact",
    ),
    (
        "WR prospect, crisp route runner with speed",
        "WR-301, 4.42 speed, 38-inch vertical, 2.1% drop rate",
    ),
])
def test_scouting_format(player, context):
    output = generator.generate(player, context)
    test_case = LLMTestCase(input=player, actual_output=output)
    format_metric = ScoutingReportFormatMetric(threshold=0.6)
    assert_test(test_case, [format_metric])


@pytest.mark.parametrize("player", [
    "QB prospect with strong arm but questionable decision-making",
    "RB prospect with speed but concerns about pass protection",
    "WR prospect with elite routes but struggles against press coverage",
])
def test_weakness_inclusion(player):
    output = generator.generate(player)
    test_case = LLMTestCase(input=player, actual_output=output)
    weakness_metric = WeaknessInclusionMetric(threshold=0.5)
    assert_test(test_case, [weakness_metric])
```

Run it:

```bash
pytest test_custom_metrics.py -v
```

Each test generates a scouting report, runs it through your custom metric, and passes or fails based on the threshold. You can see which specific checks failed in the output.

---

## Part 6: The Full Test Suite

### Step 10: One file that runs everything

Let's consolidate into a comprehensive test suite:

```python
# 11-testing-with-deepeval/test_suite.py
"""
Complete test suite for NFL scouting report quality.

Run with:  pytest test_suite.py -v
"""

import pytest
import re
from report_generator import ScoutingReportGenerator
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from custom_metrics import ScoutingReportFormatMetric, WeaknessInclusionMetric

generator = ScoutingReportGenerator()


# =========================================================
# STRUCTURE TESTS -- fast, no LLM judge
# =========================================================

class TestStructure:
    """Every scouting report must meet basic structural requirements."""

    @pytest.fixture(scope="class")
    def sample_output(self):
        return generator.generate(
            "QB prospect, pocket passer with elite accuracy",
            context="QB-101, 68% completion rate, 62 mph arm strength",
        )

    def test_has_numbered_steps(self, sample_output):
        steps = re.findall(r'^\s*\d+[\.\)]', sample_output, re.MULTILINE)
        assert len(steps) >= 3

    def test_word_count_in_range(self, sample_output):
        words = len(sample_output.split())
        assert 30 <= words <= 200

    def test_uses_scouting_language(self, sample_output):
        terms = [
            "accuracy", "arm", "release", "completion", "pocket",
            "routes", "strength", "weakness", "projection", "draft",
        ]
        found = [t for t in terms if t in sample_output.lower()]
        assert len(found) >= 2

    def test_not_excessively_verbose(self, sample_output):
        # Guard against wordy, bureaucratic output
        sentences = sample_output.split(".")
        long_sentences = [s for s in sentences if len(s.split()) > 30]
        assert len(long_sentences) <= 2, "Too many overly long sentences"


# =========================================================
# COMPLETENESS TESTS -- non-negotiable for scouting
# =========================================================

class TestCompleteness:
    """Weakness and measurable content must be present in reports."""

    @pytest.mark.parametrize("player,must_mention_one_of", [
        (
            "QB prospect, pocket passer under pressure",
            ["weakness", "concern", "struggles", "limitation", "needs work"],
        ),
        (
            "RB prospect, explosive runner",
            ["weakness", "concern", "pass protection", "blocking", "limitation"],
        ),
        (
            "WR prospect, route runner with speed",
            ["weakness", "concern", "press", "drops", "limitation"],
        ),
    ])
    def test_report_includes_weaknesses(self, player, must_mention_one_of):
        output = generator.generate(player).lower()
        matches = [t for t in must_mention_one_of if t in output]
        assert matches, (
            f"COMPLETENESS GAP: '{player}' mentions none of {must_mention_one_of}"
        )


# =========================================================
# CONSISTENCY TESTS -- reproducibility matters
# =========================================================

class TestConsistency:

    def test_same_input_same_output(self):
        player = "QB prospect, pocket passer with elite accuracy"
        a = generator.generate(player)
        b = generator.generate(player)
        assert a == b, "Non-deterministic output at temperature 0"

    def test_all_positions_have_steps(self):
        players = [
            "QB prospect, strong-armed passer",
            "RB prospect, explosive runner",
            "WR prospect, elite route runner",
        ]
        for player in players:
            output = generator.generate(player)
            steps = re.findall(r'^\s*\d+[\.\)]', output, re.MULTILINE)
            assert len(steps) >= 3, f"'{player}' only has {len(steps)} steps"


# =========================================================
# CUSTOM METRIC TESTS -- using DeepEval framework
# =========================================================

class TestCustomMetrics:

    @pytest.mark.parametrize("player,context", [
        ("QB prospect, pocket passer", "QB-101, 68% completion"),
        ("RB prospect, explosive runner", "RB-201, 4.38 40-yard dash"),
    ])
    def test_format_metric(self, player, context):
        output = generator.generate(player, context)
        test_case = LLMTestCase(input=player, actual_output=output)
        assert_test(test_case, [ScoutingReportFormatMetric(threshold=0.6)])

    @pytest.mark.parametrize("player", [
        "QB prospect with questionable pocket awareness",
        "RB prospect with speed but blocking concerns",
    ])
    def test_weakness_metric(self, player):
        output = generator.generate(player)
        test_case = LLMTestCase(input=player, actual_output=output)
        assert_test(test_case, [WeaknessInclusionMetric(threshold=0.5)])
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

**Test the things that matter.** For football scouting: weakness identification, measurables, stat support, specific references. Don't test for exact wording -- test for structural and analytical properties.

---

## Takeaways

1. **LLM tests work exactly like regular pytest tests** -- your team already knows how to use them.
2. **Heuristic tests are free and fast** -- use them for format, structure, and completeness checks on every output.
3. **Custom metrics encode YOUR quality standards** -- not generic ones from a framework.
4. **Parametrized tests catch class-wide failures** -- all QB reports need weakness assessments, not just the one you happened to test.
5. **Same input at temp=0 should give same output** -- if it doesn't, investigate.
6. **CI/CD integration is just `pytest`** -- nothing special needed.

## What's Next

You can now test and evaluate. But in production, you need **observability** -- seeing what is happening in real time, tracking performance over time, debugging failures, and managing prompt versions. Module 12 introduces **Langfuse**, an open-source observability platform that gives you a production-grade dashboard for your LLM system.
