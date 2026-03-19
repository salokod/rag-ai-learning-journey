# Module 14: Fine-Tuning

## Goal
Understand when and how to fine-tune LLMs. Learn LoRA/QLoRA techniques that make fine-tuning practical on your M4 Pro. Know when fine-tuning is the right choice vs. RAG + prompting.

---

## Concepts

### RAG vs. Fine-Tuning: When to Use Which

| Situation | Use RAG | Use Fine-Tuning |
|-----------|---------|-----------------|
| Need to reference specific documents | ✓ | |
| Need citations/sources | ✓ | |
| Data changes frequently | ✓ | |
| Need a specific writing STYLE | | ✓ |
| Need consistent FORMAT/structure | | ✓ |
| Need to learn domain TERMINOLOGY | | ✓ |
| Budget-conscious | ✓ | |

**For your manufacturing task descriptions:** Start with RAG (Modules 05-08). If the style/format still isn't matching after good prompting, fine-tune on your golden dataset.

### What Is Fine-Tuning?

Fine-tuning takes a pre-trained model and **continues training** it on your specific data. It's like hiring an experienced writer and training them on your company's style guide.

### LoRA: Making Fine-Tuning Practical

Full fine-tuning updates ALL model parameters — requires massive GPU memory. **LoRA** (Low-Rank Adaptation) freezes most parameters and adds small trainable "adapters." This reduces memory from hundreds of GB to a few GB.

```
Full Fine-Tuning:     Update 8 billion parameters → needs 80GB+ VRAM
LoRA Fine-Tuning:     Update ~8 million parameters → needs 8-16GB VRAM ✓
QLoRA (Quantized):    Same as LoRA but quantized → needs 4-8GB VRAM ✓✓
```

Your M4 Pro with 48GB unified memory can handle LoRA fine-tuning of 7-8B models comfortably.

---

## Exercise 1: Preparing Fine-Tuning Data

```python
# 14-fine-tuning/ex1_prepare_data.py
"""Prepare your manufacturing task descriptions for fine-tuning."""

import json

# Fine-tuning data format: instruction/input/output triples
# This is the format expected by most fine-tuning frameworks
training_data = [
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Inspect welded joints on Frame Assembly A. Context: AWS D1.1, Form QC-107",
        "output": """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete on the welding station.
2. Don required PPE: safety glasses, leather gloves, inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6 — check for cracks, porosity, and undercut.
4. Measure weld size with fillet gauge — minimum 6mm leg per drawing.
5. Record findings on Form QC-107. Tag defective joints with red HOLD tag and notify supervisor."""
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Perform daily forklift pre-operation inspection. Context: OSHA 1910.178, SOP-FL-001",
        "output": """DAILY FORKLIFT PRE-OPERATION INSPECTION

1. Check tire condition and inflation pressure visually.
2. Test horn, headlights, backup alarm, and strobe light.
3. Verify hydraulic fluid level — add if below MIN mark.
4. Inspect mast chains for wear, damage, or excessive slack.
5. Test service brake and parking brake before loading.
6. Record results on daily checklist per SOP-FL-001. Do not operate if any item fails."""
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Set up CNC lathe for shaft machining. Context: Drawing SH-4402-Rev.B, tolerance ±0.005\"",
        "output": """SET UP CNC LATHE FOR PRECISION SHAFT MACHINING

1. Review drawing SH-4402-Rev.B and verify material is staged.
2. Load program from DNC server — verify program number matches setup sheet.
3. Install chuck with soft jaws bored to part diameter +0.002".
4. Set tool offsets using tool setter. Verify against setup sheet (±0.001").
5. Run first article at 50% rapid, 75% feed override.
6. Record measurements on FAIR form. Proceed after QC approval."""
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Replace hydraulic cylinder seals. Context: 200-ton press, Seal kit HK-200-SEAL, SOP-SAFE-001",
        "output": """REPLACE HYDRAULIC CYLINDER SEALS

1. Perform lockout/tagout per SOP-SAFE-001. Bleed residual hydraulic pressure.
2. Disconnect hydraulic lines and cap open ports to prevent contamination.
3. Remove cylinder using overhead crane (rated capacity >2 ton). Wear hard hat.
4. Disassemble cylinder, remove old seals. Inspect bore and rod for scoring.
5. Install new seals from kit HK-200-SEAL. Lubricate with clean hydraulic fluid.
6. Reassemble, reinstall, reconnect. Bleed air from circuit.
7. Remove LOTO, pressurize slowly. Check for leaks at 0%, 50%, 100%. Log on PM-105."""
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Calibrate digital caliper. Context: NIST-traceable gauge blocks, SOP-CAL-003, Form CAL-201",
        "output": """CALIBRATE DIGITAL CALIPER

1. Clean caliper jaws and gauge blocks with lint-free cloth and isopropyl alcohol.
2. Zero the caliper with jaws fully closed — verify display reads 0.000".
3. Measure gauge blocks at 0.500", 1.000", 2.000", and 4.000".
4. Record all readings on Form CAL-201. Tolerance: ±0.001" at each point.
5. If out of tolerance, adjust per manufacturer instructions and re-test.
6. Apply calibration sticker with date, tech ID, and next-due date."""
    },
]

# Save in different formats

# Format 1: JSONL (one JSON per line) — most common for fine-tuning
with open("14-fine-tuning/training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

# Format 2: Chat format (for chat model fine-tuning)
chat_format = []
for item in training_data:
    chat_format.append({
        "messages": [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]},
        ]
    })

with open("14-fine-tuning/training_data_chat.jsonl", "w") as f:
    for item in chat_format:
        f.write(json.dumps(item) + "\n")

print(f"✓ Created training data: {len(training_data)} examples")
print(f"  JSONL format: 14-fine-tuning/training_data.jsonl")
print(f"  Chat format:  14-fine-tuning/training_data_chat.jsonl")

print("\n=== Fine-Tuning Data Guidelines ===")
print("1. QUALITY > QUANTITY: 50 perfect examples > 5000 mediocre ones")
print("2. Match your production distribution (departments, difficulty, types)")
print("3. Include edge cases (short tasks, safety-critical, multi-step)")
print("4. Have domain experts review EVERY example")
print("5. For manufacturing: minimum 20-50 examples to see style adaptation")
```

---

## Exercise 2: Fine-Tuning with Hugging Face + LoRA

```python
# 14-fine-tuning/ex2_lora_finetune.py
"""Fine-tune a model with LoRA on your manufacturing data.
This runs on your M4 Pro! MPS (Metal Performance Shaders) accelerates training."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using Apple Metal GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✓ Using NVIDIA GPU")
else:
    device = "cpu"
    print("⚠️ Using CPU (will be slow)")

# Load a small model for fine-tuning
MODEL_NAME = "microsoft/phi-2"  # 2.7B params — fits easily on M4 Pro
# Alternative: "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for even faster training

print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Half precision for memory efficiency
    trust_remote_code=True,
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                # Rank of the adaptation — higher = more capacity but more memory
    lora_alpha=32,       # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Show trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# Prepare dataset
def format_prompt(example):
    """Format training data into the prompt template."""
    text = f"""### Instruction: {example['instruction']}

### Input: {example['input']}

### Output: {example['output']}"""
    return {"text": text}

# Load and format training data
dataset = load_dataset("json", data_files="14-fine-tuning/training_data.jsonl", split="train")
dataset = dataset.map(format_prompt)

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="14-fine-tuning/output",
    num_train_epochs=3,            # Small dataset = more epochs
    per_device_train_batch_size=1, # Small batch for limited memory
    gradient_accumulation_steps=4, # Effective batch size = 4
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,                    # MPS doesn't support fp16 training well
    bf16=False,
    use_mps_device=(device == "mps"),
)

print("\n=== Ready to Fine-Tune ===")
print("To actually train, uncomment the Trainer code below.")
print("Training 5 examples for 3 epochs takes ~5-15 minutes on M4 Pro.")
print()
print("In production, you'd want:")
print("  - 50-200 training examples")
print("  - 3-5 epochs")
print("  - Validation split to detect overfitting")
print("  - Evaluation against your golden dataset after training")

# Uncomment to train:
# from transformers import Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized,
# )
# trainer.train()
# model.save_pretrained("14-fine-tuning/output/final")
# tokenizer.save_pretrained("14-fine-tuning/output/final")
```

---

## Exercise 3: Evaluating Fine-Tuned vs. Base Model

```python
# 14-fine-tuning/ex3_evaluate_finetuned.py
"""Compare fine-tuned model against base model using your evaluation pipeline."""

import ollama
import json
import re

# This exercise demonstrates the evaluation framework.
# After fine-tuning (Exercise 2), you'd load your fine-tuned model.
# For now, we'll simulate by comparing different prompt strategies.

def evaluate_output(text: str) -> dict:
    """Quick evaluation of task description quality."""
    return {
        "has_steps": len(re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)) >= 3,
        "has_safety": any(w in text.lower() for w in ["ppe", "safety", "lockout", "gloves"]),
        "has_refs": bool(re.search(r'[A-Z]{2,}-\d{2,}', text)),
        "has_action_verbs": any(w in text.lower() for w in
            ["inspect", "verify", "check", "install", "record", "measure"]),
        "word_count": len(text.split()),
    }


# Load test cases from golden dataset
with open("13-evaluation-datasets-and-benchmarks/golden_dataset.json") as f:
    golden = json.load(f)

# Compare: "base model" (minimal prompt) vs "optimized" (our best prompt)
approaches = {
    "base_model_minimal": "Write a task description.",
    "optimized_prompt": """You are a senior manufacturing technical writer at an ISO 9001 facility.
Write task descriptions with 3-7 numbered steps. Start each with an action verb.
Include PPE/safety. Reference specific forms and specifications. Active voice. Under 150 words.""",
}

results = {approach: [] for approach in approaches}

for task in golden[:3]:  # Test on first 3 golden examples
    for approach_name, system_prompt in approaches.items():
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task['task_name']}\nContext: {task['context']}"},
            ],
            options={"temperature": 0.0},
        )
        output = response["message"]["content"]
        scores = evaluate_output(output)
        scores["task"] = task["task_name"]
        results[approach_name].append(scores)

# Summary comparison
print("=== Model Comparison ===\n")
for approach_name, task_results in results.items():
    steps_pct = sum(r["has_steps"] for r in task_results) / len(task_results)
    safety_pct = sum(r["has_safety"] for r in task_results) / len(task_results)
    refs_pct = sum(r["has_refs"] for r in task_results) / len(task_results)

    print(f"{approach_name}:")
    print(f"  Steps:  {steps_pct:.0%}")
    print(f"  Safety: {safety_pct:.0%}")
    print(f"  Refs:   {refs_pct:.0%}")
    print()

print("=== When Fine-Tuning Helps ===")
print("If the 'optimized_prompt' still doesn't match your exact style,")
print("fine-tuning teaches the model your format at a deeper level.")
print("After fine-tuning, even a MINIMAL prompt produces styled output.")
print("\nThe evaluation pipeline (Modules 09-13) tells you whether")
print("fine-tuning actually improved things or was wasted effort.")
```

---

## Takeaways

1. **Try RAG + prompt engineering first** — it's cheaper, faster, and more flexible than fine-tuning
2. **Fine-tune for STYLE, not KNOWLEDGE** — RAG handles knowledge, fine-tuning handles behavior
3. **LoRA makes fine-tuning practical** — trains <1% of parameters with minimal memory
4. **Your M4 Pro can fine-tune 7B models** — no cloud GPU required for learning
5. **Always evaluate before AND after** — fine-tuning without evaluation is blind guessing

## Setting the Stage for Module 15

Fine-tuning changes HOW the model generates. Module 15 explores **agents and tool use** — giving the LLM the ability to take actions: look up specs, query databases, run calculations, and chain multiple steps together autonomously.
