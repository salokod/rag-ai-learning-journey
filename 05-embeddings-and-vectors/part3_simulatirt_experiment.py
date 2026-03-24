from openai import OpenAI
import numpy as np


llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

def embed(text):
    return llm.embeddings.create(model="nomic-embed-text", input=text).data[0].embedding

sentences = {
    "qb_accuracy": "Quarterback with elite accuracy on intermediate throws",
    "qb_precision": "Passer who delivers the ball with pinpoint precision downfield",
    "rb_vision": "Running back with exceptional vision and patience at the line",
    "unrelated": "Season ticket pricing and stadium parking availability",
    "wr_speed": "Wide receiver with 4.3 speed and deep threat ability",
    "wr_fast": "Speedy pass catcher who stretches the field vertically",
}

vecs = {name: embed(text) for name, text in sentences.items()}


def cosine_sim(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"qb_accuracy vs qb_precision: {cosine_sim(vecs['qb_accuracy'], vecs['qb_precision']):.3f}")
print(f"qb_accuracy vs unrelated: {cosine_sim(vecs['qb_accuracy'], vecs['unrelated']):.3f}")
print(f"wr_speed vs wr_fast: {cosine_sim(vecs['wr_speed'], vecs['wr_fast']):.3f}")
print(f"rb_vision vs wr_speed: {cosine_sim(vecs['rb_vision'], vecs['wr_speed']):.3f}")
