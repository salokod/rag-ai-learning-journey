import ollama

prompt = "List 10 steps for a machine startup checklist."

for penalty in [1.0, 1.2, 1.5]:
    response = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "repeat_penalty": penalty},
    )
    print(f"\n--- repeat_penalty={penalty} ---")
    print(response["message"]["content"][:400])