from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the library requires something
)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "Describe the task 'Inspect hydraulic press seals' as JSON."}],
    response_format={"type": "json_object"},
)
# print(response.choices[0].message.content)

result = json.loads(response.choices[0].message.content)
print(type(result))
print(json.dumps(result, indent=2))