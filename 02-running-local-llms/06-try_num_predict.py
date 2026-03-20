import ollama

response = ollama.chat(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "Explain lockout/tagout procedures."}],
    options={"num_predict": 50},
)

print(response["message"]["content"])
print(f"\n(Output was limited to ~50 tokens)")