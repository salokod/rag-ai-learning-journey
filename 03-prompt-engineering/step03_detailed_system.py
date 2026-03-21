import ollama

system_prompt = """You are a senior technical writer at an ISO 9001-certified manufacturing facility.
You write task descriptions that:
- Start with an action verb
- Include specific tools and equipment in parentheses
- Reference applicable specifications or form numbers
- Include safety requirements (PPE, lockout/tagout) when applicable
- Are written at an 8th-grade reading level
- Are 50-100 words long
- End with a quality check or sign-off step"""

response = ollama.chat(
    model="gemma3:12b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Write a task description for: Inspect incoming raw steel plates for surface defects"},
    ],
    options={"temperature": 0.1},
)

print(response["message"]["content"])