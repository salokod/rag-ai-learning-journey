from openai import OpenAI
import numpy as np


llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def embed(text):
    return llm.embeddings.create(model="nomic-embed-text", input=text).data[0].embedding

a = embed("Elite pocket passer with a quick release")
b = embed("Accurate quarterback who gets the ball out fast")
c = embed("Season ticket pricing and stadium parking availability")


def cosine_sim(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(cosine_sim(a, c))
