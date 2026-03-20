# Module 14: Fine-Tuning

## Goal
Understand when to fine-tune vs. use RAG + prompts. Prepare training data, set up LoRA fine-tuning, and know how to evaluate whether fine-tuning actually helped.

---

## The Key Question

Before we write a single line of code, let's answer the question that matters:

**When should you fine-tune instead of using RAG + prompts?**

Here is the framework. It is simple:

- **RAG is for knowledge.** "What is the torque spec for M10 bolts?" -- that is a RAG question. The answer lives in a document. You retrieve it.
- **Fine-tuning is for style and behavior.** "Always write numbered steps starting with action verbs, include PPE, reference forms by ID, use active voice at an 8th-grade reading level" -- that is a style requirement. The model needs to learn HOW to write, not WHAT to write.

Let's make it concrete:

| Situation | Use RAG | Use Fine-Tuning |
|-----------|---------|-----------------|
| Need specific document references | yes | |
| Need citations/sources | yes | |
| Data changes frequently | yes | |
| Need a specific writing STYLE | | yes |
| Need consistent output FORMAT | | yes |
| Need domain terminology baked in | | yes |
| Budget-conscious, need quick results | yes | |

**For your manufacturing task descriptions:** you probably need both. RAG to pull in the right specs and forms, fine-tuning to get the output style consistent. But start with RAG + good prompts. In most cases, that gets you 90% of the way. Fine-tune for the last 10%.

---

## Exercise 1: Preparing Fine-Tuning Data

Let's prepare data from your golden dataset (Module 13). Fine-tuning frameworks expect data in specific formats. We will build both.

```python
# 14-fine-tuning/ex1_prepare_data.py
"""Prepare your golden dataset for fine-tuning."""

import json
```

### The instruction format

Most fine-tuning frameworks use instruction/input/output triples. Let's build one:

```python
example_1 = {
    "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
    "input": "Task: Inspect welded joints on Frame Assembly A. Context: AWS D1.1, Form QC-107",
    "output": """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete on the welding station.
2. Don required PPE: safety glasses, leather gloves, inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6 -- check for cracks, porosity, and undercut.
4. Measure weld size with fillet gauge -- minimum 6mm leg per drawing.
5. Record findings on Form QC-107. Tag defective joints with red HOLD tag and notify supervisor."""
}
```

Notice the structure:
- **instruction** -- what you want the model to do (this stays the same for all examples)
- **input** -- the specific task and context (this varies)
- **output** -- the perfect answer (this is what the model learns to produce)

The instruction is like a standing work order. The input is the specific job. The output is the completed work.

Let's build the full training set:

```python
training_data = [
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Inspect welded joints on Frame Assembly A. Context: AWS D1.1, Form QC-107",
        "output": """INSPECT WELDED JOINTS ON FRAME ASSEMBLY A

1. Verify lockout/tagout is complete on the welding station.
2. Don required PPE: safety glasses, leather gloves, inspection magnifier.
3. Inspect all weld joints visually per AWS D1.1 Section 6 -- check for cracks, porosity, and undercut.
4. Measure weld size with fillet gauge -- minimum 6mm leg per drawing.
5. Record findings on Form QC-107. Tag defective joints with red HOLD tag and notify supervisor."""
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Perform daily forklift pre-operation inspection. Context: OSHA 1910.178, SOP-FL-001",
        "output": """DAILY FORKLIFT PRE-OPERATION INSPECTION

1. Check tire condition and inflation pressure visually.
2. Test horn, headlights, backup alarm, and strobe light.
3. Verify hydraulic fluid level -- add if below MIN mark.
4. Inspect mast chains for wear, damage, or excessive slack.
5. Test service brake and parking brake before loading.
6. Record results on daily checklist per SOP-FL-001. Do not operate if any item fails."""
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Set up CNC lathe for shaft machining. Context: Drawing SH-4402-Rev.B, tolerance +/-0.005\"",
        "output": """SET UP CNC LATHE FOR PRECISION SHAFT MACHINING

1. Review drawing SH-4402-Rev.B and verify material is staged.
2. Load program from DNC server -- verify program number matches setup sheet.
3. Install chuck with soft jaws bored to part diameter +0.002".
4. Set tool offsets using tool setter. Verify against setup sheet (+/-0.001").
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
2. Zero the caliper with jaws fully closed -- verify display reads 0.000".
3. Measure gauge blocks at 0.500", 1.000", 2.000", and 4.000".
4. Record all readings on Form CAL-201. Tolerance: +/-0.001" at each point.
5. If out of tolerance, adjust per manufacturer instructions and re-test.
6. Apply calibration sticker with date, tech ID, and next-due date."""
    },
]
```

### Save in JSONL format

JSONL (JSON Lines) is the standard format for fine-tuning. One JSON object per line:

```python
with open("14-fine-tuning/training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(training_data)} examples to training_data.jsonl")
```

Let's peek at what the file looks like:

```python
with open("14-fine-tuning/training_data.jsonl") as f:
    first_line = f.readline()
print(f"First line: {first_line[:100]}...")
```

Each line is a complete JSON object. Tools like Hugging Face and Axolotl read this natively.

### Save in chat format too

Some models (especially chat-tuned ones) expect a different format -- a list of messages:

```python
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

print(f"Saved {len(chat_format)} examples in chat format to training_data_chat.jsonl")
```

Here is the complete file:

```python
# 14-fine-tuning/ex1_prepare_data.py
"""Prepare your golden dataset for fine-tuning."""

import json

training_data = [
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Inspect welded joints on Frame Assembly A. Context: AWS D1.1, Form QC-107",
        "output": "INSPECT WELDED JOINTS ON FRAME ASSEMBLY A\n\n1. Verify lockout/tagout is complete on the welding station.\n2. Don required PPE: safety glasses, leather gloves, inspection magnifier.\n3. Inspect all weld joints visually per AWS D1.1 Section 6 -- check for cracks, porosity, and undercut.\n4. Measure weld size with fillet gauge -- minimum 6mm leg per drawing.\n5. Record findings on Form QC-107. Tag defective joints with red HOLD tag and notify supervisor."
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Perform daily forklift pre-operation inspection. Context: OSHA 1910.178, SOP-FL-001",
        "output": "DAILY FORKLIFT PRE-OPERATION INSPECTION\n\n1. Check tire condition and inflation pressure visually.\n2. Test horn, headlights, backup alarm, and strobe light.\n3. Verify hydraulic fluid level -- add if below MIN mark.\n4. Inspect mast chains for wear, damage, or excessive slack.\n5. Test service brake and parking brake before loading.\n6. Record results on daily checklist per SOP-FL-001. Do not operate if any item fails."
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Set up CNC lathe for shaft machining. Context: Drawing SH-4402-Rev.B, tolerance +/-0.005\"",
        "output": "SET UP CNC LATHE FOR PRECISION SHAFT MACHINING\n\n1. Review drawing SH-4402-Rev.B and verify material is staged.\n2. Load program from DNC server -- verify program number matches setup sheet.\n3. Install chuck with soft jaws bored to part diameter +0.002\".\n4. Set tool offsets using tool setter. Verify against setup sheet (+/-0.001\").\n5. Run first article at 50% rapid, 75% feed override.\n6. Record measurements on FAIR form. Proceed after QC approval."
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Replace hydraulic cylinder seals. Context: 200-ton press, Seal kit HK-200-SEAL, SOP-SAFE-001",
        "output": "REPLACE HYDRAULIC CYLINDER SEALS\n\n1. Perform lockout/tagout per SOP-SAFE-001. Bleed residual hydraulic pressure.\n2. Disconnect hydraulic lines and cap open ports to prevent contamination.\n3. Remove cylinder using overhead crane (rated capacity >2 ton). Wear hard hat.\n4. Disassemble cylinder, remove old seals. Inspect bore and rod for scoring.\n5. Install new seals from kit HK-200-SEAL. Lubricate with clean hydraulic fluid.\n6. Reassemble, reinstall, reconnect. Bleed air from circuit.\n7. Remove LOTO, pressurize slowly. Check for leaks at 0%, 50%, 100%. Log on PM-105."
    },
    {
        "instruction": "Write a manufacturing task description with numbered steps, safety requirements, and specification references.",
        "input": "Task: Calibrate digital caliper. Context: NIST-traceable gauge blocks, SOP-CAL-003, Form CAL-201",
        "output": "CALIBRATE DIGITAL CALIPER\n\n1. Clean caliper jaws and gauge blocks with lint-free cloth and isopropyl alcohol.\n2. Zero the caliper with jaws fully closed -- verify display reads 0.000\".\n3. Measure gauge blocks at 0.500\", 1.000\", 2.000\", and 4.000\".\n4. Record all readings on Form CAL-201. Tolerance: +/-0.001\" at each point.\n5. If out of tolerance, adjust per manufacturer instructions and re-test.\n6. Apply calibration sticker with date, tech ID, and next-due date."
    },
]

# Save JSONL (instruction format)
with open("14-fine-tuning/training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")
print(f"Saved {len(training_data)} examples to training_data.jsonl")

# Save JSONL (chat format)
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
print(f"Saved {len(chat_format)} examples in chat format to training_data_chat.jsonl")

print("\nData quality checklist:")
print("  - Every example written/reviewed by a domain expert? Check.")
print("  - Consistent style across all examples? Check.")
print("  - Mix of departments and difficulty levels? Check.")
print("  - At least 5 examples (20-50 is better for real training)? We have 5 to start.")
```

---

## Exercise 2: Understanding LoRA

Before we set up fine-tuning, let's understand what LoRA does. This is important because it is what makes fine-tuning possible on your MacBook instead of requiring a $10,000 GPU server.

```python
# 14-fine-tuning/ex2_lora_explained.py
"""Understand LoRA by looking at the numbers."""
```

Here is the core idea:

```python
# A model like Llama 3.1 8B has about 8 billion parameters.
# Full fine-tuning means updating ALL 8 billion. That requires:
total_params = 8_000_000_000
bytes_per_param_full = 4  # float32
full_memory_gb = (total_params * bytes_per_param_full) / (1024**3)
print(f"Full fine-tuning memory: {full_memory_gb:.0f} GB")
# That's ~32 GB just for the model weights, plus optimizer states = ~100+ GB total.
# Your M4 Pro has 48 GB. Not going to work.
```

Now LoRA:

```python
# LoRA freezes the original weights and adds small "adapter" matrices.
# Instead of updating a 4096x4096 weight matrix directly,
# it learns two small matrices: A (4096x16) and B (16x4096).
# The "16" is the rank (r) -- a tunable parameter.

original_matrix = 4096 * 4096  # 16,777,216 parameters
lora_a = 4096 * 16            # 65,536 parameters
lora_b = 16 * 4096            # 65,536 parameters
lora_total = lora_a + lora_b  # 131,072 parameters

reduction = lora_total / original_matrix
print(f"\nOriginal matrix: {original_matrix:,} parameters")
print(f"LoRA matrices:   {lora_total:,} parameters")
print(f"Reduction:       {reduction:.2%} of the original")
```

Run that. You will see that LoRA trains less than 1% of the parameters in each layer. Across the whole model:

```python
# Typical LoRA setup for an 8B model
lora_params = 8_000_000  # ~8 million trainable
total_params = 8_000_000_000
ratio = lora_params / total_params
print(f"\nLoRA trains {lora_params:,} out of {total_params:,} parameters")
print(f"That is {ratio:.3%} of the model")
print(f"Memory needed: ~8-16 GB (fits on your M4 Pro with room to spare)")
```

Think of it this way. You have a veteran machinist who knows everything about running a lathe. You do not need to retrain them from scratch to make parts for your company. You just need to teach them YOUR part numbering system, YOUR documentation style, YOUR inspection procedures. That is LoRA -- you are not retraining the whole model, just teaching it your specific habits.

### Setting up LoRA with Hugging Face + PEFT

Let's walk through the setup. We will build it piece by piece and explain each parameter.

```python
# 14-fine-tuning/ex2_lora_explained.py (continued)
"""LoRA fine-tuning setup with Hugging Face + PEFT."""

# NOTE: You need these packages:
# pip install torch transformers peft datasets accelerate

import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

Pick a model. We are using a smaller one so it trains fast:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Why TinyLlama? 1.1B params = fast training, low memory.
# For production, you would use Llama 3.1 8B or Phi-3.
# But for learning the process, smaller is better.

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
print("Model loaded.")
```

Now configure LoRA. Each parameter matters:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                          # Rank. Higher = more capacity, more memory. 8-64 is typical.
    lora_alpha=32,                 # Scaling factor. Usually 2x the rank.
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt. q and v attention projections.
    lora_dropout=0.05,             # Dropout for regularization. Prevents overfitting.
    bias="none",                   # Don't train bias terms.
    task_type="CAUSAL_LM",        # We're fine-tuning a text generation model.
)
```

Let me explain the key ones:

- **r=16**: This is the rank from our math above. The "16" in the 4096x16 matrices. Start with 16, increase to 32 or 64 if quality is not good enough.
- **target_modules**: Which layers in the model get LoRA adapters. "q_proj" and "v_proj" are the query and value projections in the attention mechanism. These have the most impact on output style.
- **lora_alpha=32**: A scaling factor. The rule of thumb is 2x the rank. This controls how much the LoRA adapters influence the output.

Apply it to the model:

```python
model = get_peft_model(model, lora_config)

# Let's see the numbers
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable:,}")
print(f"Total parameters:     {total:,}")
print(f"Trainable percentage: {100 * trainable / total:.2f}%")
```

Run this. You will see something like "Trainable: 4,194,304 / 1,100,000,000 (0.38%)". Less than half a percent of the model is being trained. That is LoRA.

### The training setup

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="14-fine-tuning/output",
    num_train_epochs=3,              # 3 passes through the data. Small dataset = more epochs.
    per_device_train_batch_size=1,   # Process 1 example at a time (memory-friendly).
    gradient_accumulation_steps=4,   # Accumulate 4 steps before updating = effective batch of 4.
    learning_rate=2e-4,              # Standard for LoRA. Higher than full fine-tuning.
    weight_decay=0.01,               # Light regularization.
    logging_steps=1,                 # Log every step so you can watch progress.
    save_strategy="epoch",           # Save a checkpoint after each epoch.
    fp16=False,                      # MPS doesn't support fp16 training well.
    bf16=False,                      # Same.
)

print("\nTraining configuration ready.")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Output directory: {training_args.output_dir}")
```

### Preparing the data for the trainer

```python
from datasets import load_dataset

def format_for_training(example):
    """Format an example into the prompt template the model will learn."""
    text = f"""### Instruction: {example['instruction']}

### Input: {example['input']}

### Output: {example['output']}"""
    return {"text": text}

# Load and format
dataset = load_dataset("json", data_files="14-fine-tuning/training_data.jsonl", split="train")
dataset = dataset.map(format_for_training)

def tokenize(example):
    result = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
print(f"\nDataset ready: {len(tokenized)} examples, tokenized to max 512 tokens.")
```

### Actually running training

Here is the thing: actually running training takes time. With 5 examples and 3 epochs on TinyLlama, it is about 5-10 minutes on your M4 Pro. With a larger model and more data, it could be hours.

The code below is ready to run but commented out so you can review everything first:

```python
# To actually train, uncomment this block:
#
# from transformers import Trainer
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized,
# )
#
# print("Starting training...")
# trainer.train()
#
# # Save the LoRA adapter (not the full model -- just the small adapter weights)
# model.save_pretrained("14-fine-tuning/output/lora-adapter")
# tokenizer.save_pretrained("14-fine-tuning/output/lora-adapter")
# print("Training complete. Adapter saved to 14-fine-tuning/output/lora-adapter")

print("\n=== What happens when you train ===")
print("1. The model processes each example 3 times (3 epochs)")
print("2. Loss should decrease from ~2-3 down to ~0.5-1.0")
print("3. Only the LoRA adapter weights update -- the base model is frozen")
print("4. The saved adapter is tiny (~10-50 MB) vs the full model (several GB)")
print("5. To use it: load the base model, then load the adapter on top")
```

If you want to try training for real, uncomment that block and run it. Watch the loss values -- they should go down over the epochs. If loss stops decreasing or goes up, you might be overfitting (too many epochs for too little data).

---

## Exercise 3: Evaluating Fine-Tuned vs. Base Model

This is the most important exercise. Fine-tuning without evaluation is a waste of time. You need to know: did it actually help?

We will use the evaluation approach from Module 09 to compare. Even if you did not run the actual training, this exercise shows the framework -- and you can simulate the comparison using different prompts as a proxy.

```python
# 14-fine-tuning/ex3_evaluate.py
"""Compare fine-tuned model behavior against base model."""

import ollama
import json
import re


def evaluate_output(text: str) -> dict:
    """Score a task description on key quality dimensions."""
    scores = {}
    steps = re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE)
    scores["has_steps"] = len(steps) >= 3
    scores["has_safety"] = any(w in text.lower() for w in ["ppe", "safety", "lockout", "gloves", "helmet"])
    scores["has_refs"] = bool(re.search(r'[A-Z]{2,}-\d{2,}', text))
    scores["has_action_verbs"] = any(w in text.lower() for w in ["inspect", "verify", "check", "install", "record", "measure"])
    scores["good_length"] = 30 <= len(text.split()) <= 150
    scores["overall"] = sum(float(v) for v in scores.values()) / len(scores)
    return scores
```

Now let's compare two approaches. We will use a minimal prompt (simulating the base model -- no fine-tuning, no help) vs. an optimized prompt (simulating what good prompt engineering gives you):

```python
# Load test cases
with open("13-evaluation-datasets-and-benchmarks/golden_dataset.json") as f:
    golden = json.load(f)

approaches = {
    "base_minimal": "Write a task description.",
    "prompt_engineered": """You are a senior manufacturing technical writer at an ISO 9001 facility.
Write task descriptions with 3-7 numbered steps. Start each with an action verb.
Include PPE/safety. Reference specific forms and specs. Active voice. Under 150 words.""",
}
```

Run both approaches against your golden dataset:

```python
print("=== Running Comparison ===\n")
results = {}

for approach_name, system_prompt in approaches.items():
    approach_scores = []
    for task in golden:
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task['task_name']}\nContext: {task['context']}"},
            ],
            options={"temperature": 0.0},
        )
        scores = evaluate_output(response["message"]["content"])
        scores["task_name"] = task["task_name"]
        approach_scores.append(scores)

    results[approach_name] = approach_scores

    # Print results for this approach
    avg = sum(s["overall"] for s in approach_scores) / len(approach_scores)
    print(f"{approach_name}: avg {avg:.0%}")
    for s in approach_scores:
        status = "PASS" if s["overall"] >= 0.6 else "FAIL"
        print(f"  [{status}] {s['task_name'][:45]:45s} {s['overall']:.0%}")
    print()
```

Now the comparison:

```python
# Compare
base_avg = sum(s["overall"] for s in results["base_minimal"]) / len(results["base_minimal"])
prompt_avg = sum(s["overall"] for s in results["prompt_engineered"]) / len(results["prompt_engineered"])

print("=== Comparison ===")
print(f"  Base (minimal prompt):    {base_avg:.0%}")
print(f"  Prompt-engineered:        {prompt_avg:.0%}")
print(f"  Improvement:              {prompt_avg - base_avg:+.0%}")
print()
```

Now here is the punchline:

```python
print("=== When to Fine-Tune ===")
print()
if prompt_avg >= 0.85:
    print("Your prompt-engineered score is already 85%+.")
    print("Fine-tuning MIGHT squeeze out a few more percent, but the ROI is low.")
    print("Focus on RAG retrieval quality and prompt refinement first.")
elif prompt_avg >= 0.65:
    print("Your prompt-engineered score is decent but not great.")
    print("Fine-tuning could help -- especially for consistent formatting and style.")
    print("Prepare 20-50 golden examples and try a LoRA fine-tune.")
else:
    print("Your prompt-engineered score is below 65%.")
    print("Before fine-tuning, make sure your prompts and RAG retrieval are solid.")
    print("Fine-tuning a model with bad prompts just bakes in bad habits.")

print()
print("Remember: in most cases, great prompts + RAG gets you 90% of the way.")
print("Fine-tune for the last 10% -- the consistent style, the exact format,")
print("the domain terminology that prompt engineering cannot quite nail.")
```

### After fine-tuning: how to evaluate

If you did run the training from Exercise 2, here is how you would evaluate the fine-tuned model:

```python
# After training, you would:
#
# 1. Load the base model + LoRA adapter:
#    from peft import PeftModel
#    base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#    model = PeftModel.from_pretrained(base, "14-fine-tuning/output/lora-adapter")
#
# 2. Run inference with a MINIMAL prompt (just the task, no style instructions):
#    # The whole point of fine-tuning is that the model learned the style
#    # so you should NOT need the detailed prompt anymore.
#
# 3. Compare:
#    Base model + minimal prompt:        45%  (bad)
#    Base model + engineered prompt:     78%  (good)
#    Fine-tuned model + minimal prompt:  82%  (great -- it learned the style!)
#
# That last line is the proof. The fine-tuned model produces good output
# even WITHOUT the detailed prompt. The style is baked in.

print("\n=== The Fine-Tuning Success Test ===")
print("Can your fine-tuned model produce good output with just 'Write a task description'?")
print("If yes -- fine-tuning worked. The model learned your style.")
print("If no -- you need more/better training data, or the model is too small.")
```

---

## Takeaways

1. **RAG is for knowledge, fine-tuning is for style** -- know which problem you are solving
2. **LoRA trains less than 1% of parameters** -- instead of 8 billion, you train 8 million, and it fits on your MacBook
3. **Data quality matters more than quantity** -- 50 expert-reviewed examples beats 5,000 sloppy ones
4. **Always evaluate before AND after** -- use Module 13's benchmark to prove fine-tuning helped
5. **Start with prompts + RAG** -- in most cases that gets you 90% of the way; fine-tune for the last 10%
6. **The success test**: can the fine-tuned model produce good output with a minimal prompt? If yes, it worked.

## What's Next

Fine-tuning changes HOW the model generates text. Module 15 explores agents and tool use -- giving the LLM the ability to take actions: look up specs in a database, run calculations, and chain steps together autonomously.
