import ollama

prompt = "Write one sentence describing a weld inspection task."

for top_p in [0.1, 0.5, 0.9]:
    r = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
        options={"top_p": top_p, "temperature": 0.8},
    )
    text = r["message"]["content"].strip()
    print(f"  top_p={top_p}: {text[:90]}")