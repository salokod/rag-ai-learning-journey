import ollama

# First call: give it information
r1 = ollama.chat(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "Remember: the spec number is WPS-201."}],
)
print(f"Response 1: {r1['message']['content'].strip()[:8000]}")

# Second call: ask about that information
r2 = ollama.chat(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "What spec number did I just mention?"}],
)
print(f"Response 2: {r2['message']['content'].strip()[:8000]}")