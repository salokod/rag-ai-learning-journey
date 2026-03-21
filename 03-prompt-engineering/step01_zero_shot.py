# 03-prompt-engineering/step01_zero_shot.py
import ollama

response = ollama.chat(
    model="gemma3:12b",
    messages=[
        {"role": "user", "content": "Write a task description for: Inspect incoming raw steel plates for surface defects"}
    ],
    options={"temperature": 0.1},
)

print(response["message"]["content"])