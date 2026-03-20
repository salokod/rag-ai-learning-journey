import ollama

examples = [
    "the",
    "inspect",
    "manufacturing",
    "WPS-201-Rev.C",
    "PPE",
]

for word in examples:
    chars = len(word)
    approx_tokens = max(1, chars // 4)
    print(f"  '{word}' -- {chars} chars, ~{approx_tokens} token(s)")