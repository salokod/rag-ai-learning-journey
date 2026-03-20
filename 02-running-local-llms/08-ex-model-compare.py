
import ollama
import time

PROMPT = """Write a professional task description for an assembly line operator
who needs to install a circuit board into a housing unit. Include safety
requirements and quality checks. Keep it under 100 words."""

models_to_test = ["gemma3:12b", "qwen3:32b"]

results = {}

for model_name in models_to_test:
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    try:
        start = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT}],
            options={"temperature": 0.1},
        )
        elapsed = time.time() - start

        content = response["message"]["content"]
        word_count = len(content.split())

        results[model_name] = {
            "content": content,
            "time": elapsed,
            "words": word_count,
        }

        print(f"Time: {elapsed:.1f}s | Words: {word_count}")
        print(f"\n{content}")

    except Exception as e:
        print(f"Error: {e}")
        print(f"(You may need to pull this model: ollama pull {model_name})")

# Summary
print(f"\n{'=' * 60}")
print("COMPARISON SUMMARY")
print(f"{'=' * 60}")
print(f"{'Model':<20} {'Time':>6} {'Words':>6}")
print(f"{'-'*20} {'-'*6} {'-'*6}")
for model_name, data in results.items():
    print(f"{model_name:<20} {data['time']:>5.1f}s {data['words']:>5d}")

print("\nLook at both outputs and ask yourself:")
print("  - Which one followed the 'under 100 words' instruction?")
print("  - Which one included safety requirements?")
print("  - Which one sounds like it belongs in a real work instruction?")
print("  - Is the faster model 'good enough' for your use case?")
print("\nThere's no single right answer. It depends on what you need.")
print("For learning and iteration: fast is better (gemma3:12b or phi3:mini).")
print("For quality output: bigger is usually better (llama3.3:70b for modules 14+).")