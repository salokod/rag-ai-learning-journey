import ollama

messages = [
    {"role": "user", "content": "Remember: the spec number is WPS-201."},
    {"role": "assistant", "content": "Got it, the spec number is WPS-201."},
    {"role": "user", "content": "What spec number did I just mention?"},
]

r = ollama.chat(model="gemma3:12b", messages=messages)
print(f"Response: {r['message']['content'].strip()[:8000]}")