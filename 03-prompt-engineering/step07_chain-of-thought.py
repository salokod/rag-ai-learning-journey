import ollama

system_prompt = """You are a manufacturing technical writer. Before writing a task description,
think through these questions:
1. What tools or equipment does the operator need?
2. What safety precautions apply (PPE, lockout/tagout)?
3. What are the quality or acceptance criteria?
4. What documentation must be completed?

Show your thinking, then write the final task description."""

response = ollama.chat(
    model="gemma3:12b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Write a task description for: Set up CNC lathe for shaft machining run"},
    ],
    options={"temperature": 0.1},
)

print(response["message"]["content"])