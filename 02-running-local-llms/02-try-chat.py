# import ollama

# response = ollama.chat(
#     model="gemma3:12b",
#     messages=[
#         {"role": "system", "content": "You are a manufacturing process engineer. Be concise and technical."},
#         {"role": "user", "content": "Name 3 types of non-destructive testing used in manufacturing."},
#     ],
# )

# print(response["message"]["content"])

import ollama

result = ollama.generate(
    model="gemma3:12b",
    prompt="Complete this sentence: The operator shall inspect each weld joint by",
)

print(result["response"])