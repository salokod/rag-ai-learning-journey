from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the library requires something
)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {"role": "system", "content": "You are a technical writer for manufacturing documentation."},
        {"role": "user", "content": "Write a safety precaution for operating a hydraulic press."},
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)