import ollama

prompt = "Write one sentence describing a weld inspection task."

print("=== Temperature 0.0 (deterministic) ===")
for i in range(3):
    r = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    print(f"  Run {i+1}: {r['message']['content'].strip()[:100]}")