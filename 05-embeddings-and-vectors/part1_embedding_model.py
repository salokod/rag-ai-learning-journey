from openai import OpenAI

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = llm.embeddings.create(model="nomic-embed-text", input="Hello world")
print(type(response.data[0].embedding))
print(len(response.data[0].embedding))
print(response.data[0].embedding[:10])
