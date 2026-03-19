# Module 19: Capstone — Manufacturing Task Description System

## Goal
Build a complete, production-grade system that generates manufacturing task descriptions using RAG, evaluates their quality, enforces safety guardrails, and provides full observability. This is everything you've learned, integrated into one system.

---

## The System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                MANUFACTURING TASK DESCRIPTION SYSTEM                │
│                                                                     │
│  ┌───────────┐    ┌──────────────┐    ┌───────────────┐            │
│  │  User      │───→│ Input        │───→│ RAG Pipeline  │            │
│  │  Request   │    │ Guardrails   │    │               │            │
│  └───────────┘    │ (Module 16)  │    │ ┌───────────┐ │            │
│                    └──────────────┘    │ │ Retrieve  │ │            │
│                                        │ │ (Mod 5-7) │ │            │
│  ┌───────────┐    ┌──────────────┐    │ └─────┬─────┘ │            │
│  │  Task      │←──│ Output       │←──│ ┌─────↓─────┐ │            │
│  │  Descr.    │    │ Guardrails   │    │ │ Generate  │ │            │
│  └───────────┘    │ (Module 16)  │    │ │ (Mod 3-4) │ │            │
│                    └──────┬───────┘    │ └───────────┘ │            │
│                           │            └───────────────┘            │
│                    ┌──────↓───────┐                                 │
│                    │ Evaluate     │    ┌───────────────┐            │
│                    │ (Mod 9-13)  │───→│ Langfuse      │            │
│                    └──────────────┘    │ (Module 12)   │            │
│                                        └───────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Build Plan

This capstone is structured in 5 stages. Each stage adds a layer of capability.

### Stage 1: Knowledge Base
### Stage 2: RAG Pipeline
### Stage 3: Evaluation Layer
### Stage 4: Guardrails
### Stage 5: API & Observability

---

## Stage 1: Build the Knowledge Base

```python
# 19-capstone-manufacturing-tasks/stage1_knowledge_base.py
"""Stage 1: Set up the manufacturing knowledge base."""

import chromadb
import json

# In a real project, these would be loaded from your actual company documents
# via the document processing pipeline (Module 08)
MANUFACTURING_KNOWLEDGE = [
    # Specifications
    {"id": "MT-302", "text": "Torque Specification MT-302 for Frame Assembly #4200. Grade 8 zinc plated fasteners. M8 bolts: 25-30 Nm. M10 bolts: 45-55 Nm. M12 bolts: 80-100 Nm. Sequence: star pattern per diagram. Tool: calibrated torque wrench ±2% accuracy. QC verification: 10% sampling. Documentation: Form QC-110.", "type": "specification", "department": "assembly"},
    {"id": "WPS-201", "text": "Welding Procedure Specification WPS-201. Process: GMAW (MIG). Base metal: Carbon steel A36. Filler: ER70S-6, 0.035 inch. Shielding gas: 75% Argon / 25% CO2 at 25-30 CFH. Preheat: not required under 1 inch. Interpass temperature: 400°F maximum. Post-weld: visual inspection required. UT for critical joints. Acceptance per AWS D1.1 Section 6.", "type": "specification", "department": "welding"},
    {"id": "SOP-CNC-042", "text": "CNC Machine Daily Startup. 1. Visual inspection of machine and area. 2. Check coolant level, refill if below MIN. 3. Check way oil level. 4. Power on, home all axes. 5. Spindle warmup O9000: 500 RPM 5 min, 2000 RPM 5 min. 6. Verify axes with test indicator. 7. Air pressure minimum 80 PSI. 8. Log on daily checklist.", "type": "SOP", "department": "machining"},

    # Safety
    {"id": "SOP-SAFE-001", "text": "Lockout/Tagout Procedure. Before maintenance: 1. Notify affected operators. 2. Normal shutdown. 3. Isolate ALL energy sources (electrical, hydraulic, pneumatic). 4. Apply personal lock and tag. 5. Release stored energy (bleed hydraulics, discharge capacitors). 6. Verify zero energy by attempting restart. After: remove tools, replace guards, verify clear, remove locks (ONLY by person who applied).", "type": "safety", "department": "all"},
    {"id": "PPE-001", "text": "PPE Requirements by Task: General production: safety glasses, steel-toe boots. Welding: auto-darkening helmet shade 10-13, leather gloves, FR clothing, safety glasses under helmet. Grinding: face shield, safety glasses, leather gloves, hearing protection, ensure guard installed. CNC machining: safety glasses, hearing protection above 85dB, no loose clothing. Press: safety glasses, steel-toe boots, hearing protection, never bypass interlocks. Forklift: hard hat when loading overhead, seatbelt.", "type": "safety", "department": "all"},

    # Quality Forms
    {"id": "QC-107", "text": "Quality Control Form QC-107: Visual and Dimensional Inspection. Required fields: part number, lot number, inspector badge ID, date, shift. Visual checklist: surface finish (no scratches/dents beyond spec), weld quality (no cracks, porosity, undercut, incomplete fusion), hardware (all fasteners present and torqued), paint/coating (uniform, no runs/drips/bare spots). Pass criteria: ALL items must pass. Failure: apply red HOLD tag, notify shift supervisor immediately.", "type": "form", "department": "quality"},
    {"id": "QC-110", "text": "Quality Control Form QC-110: Dimensional Inspection Report. Record actual measurement vs. nominal for each controlled dimension per drawing. Required: part number, operation number, inspector ID, gauge ID (must be in calibration). Flag any dimension outside tolerance in red. Requires QC supervisor signature for disposition of non-conforming parts.", "type": "form", "department": "quality"},
    {"id": "QC-115", "text": "Quality Control Form QC-115: Weld-Specific Inspection Report. References applicable WPS, joint type, weld position, welder qualification number. Visual criteria per AWS D1.1 Table 6.1. NDE results: UT, MT, PT as required by drawing. Acceptance/rejection with specific clause reference.", "type": "form", "department": "quality"},
    {"id": "CAL-201", "text": "Calibration Record Form CAL-201. Fields: instrument serial number, description, calibration standard (must be NIST-traceable), readings at each test point, pass/fail per tolerance, calibration technician ID, date performed, next due date. Calibration sticker applied to instrument.", "type": "form", "department": "metrology"},

    # Format Reference
    {"id": "FORMAT-001", "text": "Task Description Format Standard: Title in ALL CAPS on first line. Blank line. Body: 3-7 numbered steps. Each step starts with an action verb (Inspect, Verify, Install, etc.). Include tool/equipment references in parentheses. Include specification/form references. Include PPE/safety as first or second step for hazardous tasks. Final step: documentation or quality verification. Target: 50-120 words. Active voice only.", "type": "standard", "department": "all"},
]


def build_knowledge_base() -> chromadb.Collection:
    """Build and populate the manufacturing knowledge base."""
    client = chromadb.PersistentClient(path="19-capstone-manufacturing-tasks/chroma_db")

    # Delete existing collection if it exists
    try:
        client.delete_collection("manufacturing_kb")
    except ValueError:
        pass

    collection = client.create_collection(
        name="manufacturing_kb",
        metadata={"description": "Manufacturing SOPs, specs, forms, and standards"},
    )

    collection.add(
        ids=[doc["id"] for doc in MANUFACTURING_KNOWLEDGE],
        documents=[doc["text"] for doc in MANUFACTURING_KNOWLEDGE],
        metadatas=[{"type": doc["type"], "department": doc["department"]} for doc in MANUFACTURING_KNOWLEDGE],
    )

    print(f"✓ Knowledge base built: {collection.count()} documents")
    return collection


if __name__ == "__main__":
    kb = build_knowledge_base()

    # Verify with a test query
    results = kb.query(query_texts=["welding safety PPE"], n_results=3)
    print("\nTest query: 'welding safety PPE'")
    for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
        print(f"  [{doc_id}] {doc[:60]}...")
```

---

## Stage 2: RAG Task Description Generator

```python
# 19-capstone-manufacturing-tasks/stage2_rag_generator.py
"""Stage 2: RAG-powered task description generator."""

import chromadb
import ollama
from stage1_knowledge_base import build_knowledge_base

SYSTEM_PROMPT = """You are a senior manufacturing technical writer at an ISO 9001 certified facility.

Generate task descriptions following these EXACT rules:
- Title in ALL CAPS on the first line
- Blank line after title
- 3-7 numbered steps, each starting with an action verb
- Include specific tool references in parentheses
- Include specification and form numbers from the provided reference documents
- Include PPE/safety requirements (reference PPE-001 or SOP-SAFE-001 as applicable)
- Final step must be documentation or quality verification
- 50-120 words total
- Active voice ONLY

Use ONLY information from the provided reference documents. Do NOT invent specification numbers, form numbers, or procedures."""


class TaskDescriptionGenerator:
    """Generate manufacturing task descriptions using RAG."""

    def __init__(self, collection: chromadb.Collection, model: str = "llama3.1:8b"):
        self.collection = collection
        self.model = model

    def retrieve(self, task_name: str, department: str = None, n_results: int = 4) -> list[dict]:
        """Retrieve relevant documents for the task."""
        query_kwargs = {
            "query_texts": [task_name],
            "n_results": n_results,
        }
        if department:
            query_kwargs["where"] = {
                "$or": [
                    {"department": department},
                    {"department": "all"},
                ]
            }

        results = self.collection.query(**query_kwargs)
        return [
            {"id": doc_id, "text": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            )
        ]

    def generate(self, task_name: str, department: str = "", context: str = "") -> dict:
        """Generate a task description with RAG."""

        # Retrieve relevant documents
        retrieved = self.retrieve(task_name, department)
        sources = [d["id"] for d in retrieved]
        context_docs = "\n\n".join(f"[{d['id']}]: {d['text']}" for d in retrieved)

        # Additional context from user
        extra = f"\nAdditional context: {context}" if context else ""

        # Generate
        user_prompt = f"""Generate a task description for:

Task: {task_name}
Department: {department}
{extra}

REFERENCE DOCUMENTS:
{context_docs}

Write the task description now, following ALL format rules exactly."""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.1, "repeat_penalty": 1.2},
        )

        return {
            "task_name": task_name,
            "department": department,
            "description": response["message"]["content"],
            "sources": sources,
            "model": self.model,
        }


if __name__ == "__main__":
    kb = build_knowledge_base()
    generator = TaskDescriptionGenerator(kb)

    # Test with several tasks
    tasks = [
        ("Inspect welded joints on Frame Assembly A", "quality", "Per WPS-201 and AWS D1.1"),
        ("Set up CNC lathe for precision shaft", "machining", "Drawing SH-4402-Rev.B"),
        ("Perform daily forklift inspection", "warehouse", ""),
        ("Verify torque on Frame #4200 bolts", "assembly", ""),
        ("Calibrate digital micrometer", "metrology", "0-1 inch range, NIST traceable"),
    ]

    for task_name, dept, ctx in tasks:
        result = generator.generate(task_name, dept, ctx)
        print(f"\n{'='*60}")
        print(f"Task: {result['task_name']}")
        print(f"Sources: {result['sources']}")
        print(f"{'='*60}")
        print(result["description"])
```

---

## Stage 3: Evaluation Layer

```python
# 19-capstone-manufacturing-tasks/stage3_evaluation.py
"""Stage 3: Comprehensive evaluation of generated task descriptions."""

import re
import json
import ollama


class TaskDescriptionEvaluator:
    """Multi-layered evaluation for manufacturing task descriptions."""

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def heuristic_eval(self, text: str) -> dict:
        """Fast, deterministic quality checks."""
        steps = re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)
        word_count = len(text.split())
        lines = text.strip().split('\n')
        first_line_caps = lines[0].isupper() if lines else False

        checks = {
            "title_in_caps": first_line_caps,
            "has_3_plus_steps": len(steps) >= 3,
            "has_7_or_fewer_steps": len(steps) <= 7,
            "word_count_50_to_120": 50 <= word_count <= 150,  # Slight flexibility
            "has_safety_mention": any(w in text.lower() for w in
                ["ppe", "safety", "lockout", "tagout", "glasses", "gloves", "helmet"]),
            "has_spec_reference": bool(re.search(r'[A-Z]{2,}-\d{2,}', text)),
            "has_form_reference": bool(re.search(r'[Ff]orm\s+[A-Z]', text)) or "QC-" in text or "CAL-" in text,
            "uses_active_voice": sum(1 for p in ["should be", "is to be", "must be done"]
                if p in text.lower()) == 0,
            "has_action_verbs": sum(1 for v in
                ["inspect", "verify", "check", "install", "remove", "record", "don",
                 "measure", "apply", "ensure", "clean", "document", "perform"]
                if v in text.lower()) >= 3,
        }

        passed = sum(checks.values())
        total = len(checks)
        checks["score"] = round(passed / total, 2)
        checks["word_count"] = word_count
        checks["step_count"] = len(steps)
        return checks

    def llm_eval(self, text: str, task_context: str = "") -> dict:
        """Deep quality evaluation using LLM judge."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Rate this manufacturing task description 0-10 on:
1. clarity: Can an operator follow this without confusion?
2. completeness: Are all necessary steps included?
3. safety: Are relevant safety precautions mentioned?
4. specificity: Does it reference specific tools, specs, forms?
5. professionalism: Does it read like a professional manufacturing document?

Return ONLY JSON: {"clarity": N, "completeness": N, "safety": N, "specificity": N, "professionalism": N, "overall": N, "suggestions": ["..."]}""",
                },
                {"role": "user", "content": f"Context: {task_context}\n\nTask Description:\n{text}"},
            ],
            format="json",
            options={"temperature": 0.0},
        )

        try:
            scores = json.loads(response["message"]["content"])
            # Normalize to 0-1
            for key in ["clarity", "completeness", "safety", "specificity", "professionalism", "overall"]:
                if key in scores and isinstance(scores[key], (int, float)):
                    scores[key] = round(scores[key] / 10, 2)
            return scores
        except json.JSONDecodeError:
            return {"error": "parse_failed", "overall": 0.0}

    def evaluate(self, text: str, task_context: str = "") -> dict:
        """Full evaluation pipeline."""
        heuristic = self.heuristic_eval(text)
        llm = self.llm_eval(text, task_context)

        combined = round(0.4 * heuristic["score"] + 0.6 * llm.get("overall", 0), 2)

        return {
            "heuristic": heuristic,
            "llm": llm,
            "combined_score": combined,
            "passes_threshold": combined >= 0.7,
            "needs_review": combined < 0.8,
        }
```

---

## Stage 4: Guardrails

```python
# 19-capstone-manufacturing-tasks/stage4_guardrails.py
"""Stage 4: Safety guardrails for manufacturing content."""

import re


class ManufacturingGuardrails:
    """Input and output guardrails for manufacturing task descriptions."""

    VALID_REFS = {
        "MT-302", "WPS-201", "SOP-SAFE-001", "SOP-FL-001", "SOP-CAL-003",
        "SOP-CNC-042", "QC-107", "QC-110", "QC-115", "CAL-201", "PM-105",
        "PPE-001", "FORMAT-001", "AWS-D1.1",
    }

    DANGEROUS_PATTERNS = [
        (r'bypass\s+(safety|interlock|guard)', "Suggests bypassing safety"),
        (r'(skip|ignore)\s+(lockout|tagout|loto)', "Suggests skipping LOTO"),
        (r'not\s+necessary\s+to\s+wear', "Suggests PPE not needed"),
        (r'(remove|disable)\s+(guard|safety)', "Suggests removing safety"),
    ]

    def validate_input(self, task_name: str) -> dict:
        """Validate task name input."""
        issues = []
        if len(task_name) < 5:
            issues.append("Task name too short")
        if len(task_name) > 200:
            issues.append("Task name too long")

        # Check for injection
        injection_patterns = [r'ignore.*instructions', r'system\s*:', r'<script']
        for p in injection_patterns:
            if re.search(p, task_name, re.IGNORECASE):
                issues.append("Potential prompt injection detected")
                break

        return {"pass": len(issues) == 0, "issues": issues}

    def validate_output(self, text: str) -> dict:
        """Validate generated task description."""
        issues = []

        # Check for hallucinated references
        found_refs = set(re.findall(r'[A-Z]{2,3}-\d{2,}', text))
        unknown_refs = found_refs - self.VALID_REFS
        if unknown_refs:
            issues.append(f"Unverified references: {unknown_refs}")

        # Check for dangerous content
        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"SAFETY: {description}")

        return {
            "pass": len(issues) == 0,
            "issues": issues,
            "verified_refs": found_refs & self.VALID_REFS,
            "needs_human_review": len(issues) > 0,
        }
```

---

## Stage 5: Putting It All Together

```python
# 19-capstone-manufacturing-tasks/stage5_full_system.py
"""Stage 5: The complete manufacturing task description system."""

import chromadb
from stage1_knowledge_base import build_knowledge_base
from stage2_rag_generator import TaskDescriptionGenerator
from stage3_evaluation import TaskDescriptionEvaluator
from stage4_guardrails import ManufacturingGuardrails


class ManufacturingTaskSystem:
    """Complete system: input validation → RAG generation → evaluation → output validation."""

    def __init__(self, model: str = "llama3.1:8b"):
        self.kb = build_knowledge_base()
        self.generator = TaskDescriptionGenerator(self.kb, model)
        self.evaluator = TaskDescriptionEvaluator(model)
        self.guardrails = ManufacturingGuardrails()

    def generate_task_description(self, task_name: str, department: str = "",
                                   context: str = "") -> dict:
        """Full pipeline: validate → generate → evaluate → validate output."""

        # Step 1: Input validation
        input_check = self.guardrails.validate_input(task_name)
        if not input_check["pass"]:
            return {"status": "rejected", "reason": input_check["issues"]}

        # Step 2: RAG generation
        result = self.generator.generate(task_name, department, context)

        # Step 3: Output guardrails
        output_check = self.guardrails.validate_output(result["description"])

        # Step 4: Quality evaluation
        evaluation = self.evaluator.evaluate(
            result["description"],
            task_context=f"{task_name} ({department})",
        )

        # Step 5: Determine status
        if not output_check["pass"]:
            status = "needs_human_review"
        elif not evaluation["passes_threshold"]:
            status = "low_quality"
        else:
            status = "approved"

        return {
            "status": status,
            "task_description": result["description"],
            "sources": result["sources"],
            "evaluation": {
                "combined_score": evaluation["combined_score"],
                "heuristic_score": evaluation["heuristic"]["score"],
                "llm_score": evaluation["llm"].get("overall", 0),
                "suggestions": evaluation["llm"].get("suggestions", []),
            },
            "guardrails": {
                "input_passed": input_check["pass"],
                "output_passed": output_check["pass"],
                "output_issues": output_check.get("issues", []),
                "verified_refs": list(output_check.get("verified_refs", set())),
            },
        }


def main():
    """Run the complete system."""
    system = ManufacturingTaskSystem()

    test_tasks = [
        {
            "task_name": "Inspect welded joints on Frame Assembly #4200",
            "department": "quality",
            "context": "Per WPS-201, AWS D1.1 compliance required",
        },
        {
            "task_name": "Set up CNC lathe for precision shaft machining",
            "department": "machining",
            "context": "Drawing SH-4402-Rev.B, tolerance ±0.005 inch",
        },
        {
            "task_name": "Perform daily forklift pre-operation inspection",
            "department": "warehouse",
            "context": "",
        },
        {
            "task_name": "Verify torque on Frame Assembly #4200",
            "department": "assembly",
            "context": "Per specification MT-302",
        },
        {
            "task_name": "Calibrate digital micrometer",
            "department": "metrology",
            "context": "0-1 inch range, NIST-traceable standards",
        },
    ]

    print("=" * 70)
    print("  MANUFACTURING TASK DESCRIPTION SYSTEM — FULL PIPELINE")
    print("=" * 70)

    for task in test_tasks:
        result = system.generate_task_description(**task)

        status_icon = {"approved": "✓", "needs_human_review": "⚠️", "low_quality": "✗", "rejected": "🚫"}
        icon = status_icon.get(result["status"], "?")

        print(f"\n{'─'*70}")
        print(f"{icon} Task: {task['task_name']}")
        print(f"  Status: {result['status']}")
        print(f"  Score: {result['evaluation']['combined_score']:.0%}")
        print(f"  Sources: {result.get('sources', [])}")
        print(f"  Verified refs: {result['guardrails'].get('verified_refs', [])}")

        if result['guardrails'].get('output_issues'):
            print(f"  ⚠️  Issues: {result['guardrails']['output_issues']}")

        if result.get('evaluation', {}).get('suggestions'):
            print(f"  Suggestions: {result['evaluation']['suggestions']}")

        print(f"\n{result.get('task_description', 'N/A')}")

    # System summary
    print(f"\n{'='*70}")
    print("SYSTEM SUMMARY")
    print(f"{'='*70}")
    statuses = [system.generate_task_description(**t)["status"] for t in test_tasks[:2]]
    print(f"This system provides:")
    print(f"  ✓ RAG-powered generation with cited sources")
    print(f"  ✓ Multi-layer evaluation (heuristic + LLM-as-judge)")
    print(f"  ✓ Input/output guardrails with reference validation")
    print(f"  ✓ Quality scoring with pass/fail thresholds")
    print(f"  ✓ Human review flagging for borderline cases")


if __name__ == "__main__":
    main()
```

---

## Running the Capstone

```bash
# From the repo root, with venv activated
cd 19-capstone-manufacturing-tasks

# Stage by stage:
python stage1_knowledge_base.py    # Build knowledge base
python stage2_rag_generator.py     # Test RAG generation
python stage5_full_system.py       # Run the complete system
```

---

## What You've Built

| Component | Modules Used | What It Does |
|-----------|-------------|-------------|
| Knowledge Base | 05, 08 | Stores and retrieves manufacturing documents |
| RAG Generator | 03, 04, 06, 07 | Generates task descriptions grounded in company docs |
| Evaluator | 09, 10, 11, 13 | Scores quality with heuristics + LLM judge |
| Guardrails | 16 | Validates input/output, catches hallucinated refs |
| API Layer | 18 | Production-ready HTTP API with caching |
| Observability | 12 | Traces, scores, and monitors every request |

---

## Next Steps for Production

1. **Load your actual documents** — Replace sample data with real SOPs, specs, and forms
2. **Expand the golden dataset** — Get domain experts to write/review 50+ test cases
3. **Run benchmarks** — Measure against your golden dataset before each change
4. **Set up Langfuse** — Deploy the observability dashboard
5. **A/B test with humans** — Compare AI-generated vs. manually-written descriptions
6. **Iterate based on data** — Evaluation scores tell you exactly what to improve

---

## Congratulations

You now have:
- A deep understanding of **LLMs, tokens, and how generation works**
- Hands-on experience with **local models on your M4 Pro**
- **Prompt engineering** skills for consistent, professional output
- A complete **RAG pipeline** from document ingestion to generation
- A **comprehensive evaluation framework** (Ragas, DeepEval, custom metrics)
- **Langfuse observability** as your open-source Galileo alternative
- **Golden datasets and regression benchmarks** for quality assurance
- **Guardrails** that catch hallucinations and dangerous content
- **Production patterns**: APIs, caching, cost tracking, monitoring
- A **working capstone system** ready for real manufacturing data

You can walk into that meeting and say: "Here's the system, here's the evaluation data, and here's why we can trust it."
