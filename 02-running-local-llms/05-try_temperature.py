import ollama

prompt = "Describe the process of TIG welding aluminum in exactly 2 sentences."

for temp in [0.0, 0.5, 1.0, 1.5]:
    response = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temp},
    )
    print(f"\n--- temperature={temp} ---")
    print(response["message"]["content"])