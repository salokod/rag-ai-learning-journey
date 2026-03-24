from openai import OpenAI
import numpy as np


llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

def embed(text):
    return llm.embeddings.create(model="nomic-embed-text", input=text).data[0].embedding


def cosine_sim(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


doc = embed("Pocket passer with elite accuracy. Completes 68% of passes with a 2.3-second average release time. Excels on intermediate routes (15-25 yards) with anticipation throws.")
query = embed("who's a good deep ball passer")
print(f"Similarity: {cosine_sim(doc, query):.3f}")

doc2 = embed("Explosive runner with 4.38 40-yard dash. Exceptional vision through traffic and finds cutback lanes consistently.")
print(f"RB doc vs QB query: {cosine_sim(doc2, query):.3f}")
