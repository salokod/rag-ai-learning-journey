import ollama

response = ollama.chat(
    model="gemma3:12b",
    messages=[
        {"role": "system", "content": "You are a manufacturing technical writer."},
        {"role": "user", "content": "Write a task description for: Inspect incoming raw steel plates for surface defects"},
    ],
    options={"temperature": 0.1},
)

print(response["message"]["content"])