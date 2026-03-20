import ollama

stream = ollama.chat(
    model="gemma3:12b",
    messages=[
        {"role": "user", "content": "List 5 common safety hazards in a machine shop. One sentence each."}
    ],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()