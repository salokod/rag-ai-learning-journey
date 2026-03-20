import ollama

prompt = "Write one sentence describing a weld inspection task."

print("=== Temperature 1.5 (very creative) ===")
for i in range(3):
    r = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 1.5},
    )
    print(f"  Run {i+1}: {r['message']['content'].strip()[:100]}")