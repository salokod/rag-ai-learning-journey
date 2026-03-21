# 03-prompt-engineering/step08_prompt_versions.py
import ollama
import json
from datetime import datetime

# Define prompt versions -- each one is an iteration
PROMPT_VERSIONS = {
    "v1": {
        "system": "You write manufacturing task descriptions.",
        "notes": "Minimal -- just the bare instruction",
    },
    "v2": {
        "system": """You write manufacturing task descriptions.
Keep them under 100 words. Use active voice. Include safety notes.""",
        "notes": "Added basic constraints: length, voice, safety",
    },
    "v3": {
        "system": """You are a technical writer for an ISO 9001 facility.
Write task descriptions with numbered steps (3-5 steps).
Start each step with an action verb.
Include tool references in parentheses.
Final step must be documentation or verification.
Active voice. 50-100 words.""",
        "notes": "Detailed format, style, and structure constraints",
    },
    "v4": {
        "system": """You are a technical writer for an ISO 9001 facility.
Write task descriptions with numbered steps (3-5 steps), in roman numeral numbers.
Start each step with an action verb.
Include tool references in parentheses.
Final step must be documentation or verification.
Make it so readable, a cave man could read it.
Active voice. 50-100 words.""",
        "notes": "Detailed format, style, and structure constraints",
    },
}

test_input = "Write a task description for: Calibrate digital pressure gauge"

results = []

for version, config in PROMPT_VERSIONS.items():
    response = ollama.chat(
        model="gemma3:12b",
        messages=[
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": test_input},
        ],
        options={"temperature": 0.0},
    )

    output = response["message"]["content"]

    result = {
        "version": version,
        "notes": config["notes"],
        "output": output,
        "word_count": len(output.split()),
        "has_numbered_steps": any(f"{i}." in output for i in range(1, 6)),
        "timestamp": datetime.now().isoformat(),
    }
    results.append(result)

    print(f"\n{'=' * 60}")
    print(f"{version}: {config['notes']}")
    print(f"Words: {result['word_count']} | Numbered steps: {result['has_numbered_steps']}")
    print(f"{'=' * 60}")
    print(output[:400])

# Save results
with open("03-prompt-engineering/prompt_iterations.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print("Results saved to prompt_iterations.json")
print(f"{'=' * 60}")
print("\nLook at the progression:")
print("  v1 -> Generic, unstructured, inconsistent length")
print("  v2 -> Better, but format is still unpredictable")
print("  v3 -> Numbered steps, action verbs, consistent structure")
print("\nEach version is an improvement because we changed ONE thing")
print("and checked the result. This is prompt engineering as a process.")