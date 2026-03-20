import ollama

prompt = "Write one sentence describing a weld inspection task."
temps = [0.0, 0.3, 0.7, 1.0, 1.5, 4]

for temp in temps:
    r = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temp},
    )
    text = r["message"]["content"].strip()
    print(f"  temp={temp}: {text[:900]}")