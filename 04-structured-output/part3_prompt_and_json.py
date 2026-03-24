from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the library requires something
)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": """Return JSON with exactly these fields:
- task_id: string in format "TD-XXXX"
- title: short task title
- steps: array of action steps
- safety: array of safety requirements""",
        },
        {"role": "user", "content": "Task: Inspect hydraulic press cylinder seals for wear and leakage"},
    ],
    response_format={"type": "json_object"},
)

result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2))

print(list(result.keys()))

print(type(result["steps"]))
print(type(result["task_id"]))