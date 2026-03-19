"""Verify that the learning journey environment is ready."""

import sys
print(f"✓ Python {sys.version}")

# Check key packages
packages = {
    "ollama": "ollama",
    "chromadb": "chromadb",
    "langchain": "langchain",
    "ragas": "ragas",
    "deepeval": "deepeval",
    "langfuse": "langfuse",
    "transformers": "transformers",
    "peft": "peft",
    "fastapi": "fastapi",
    "rich": "rich",
}

for display_name, import_name in packages.items():
    try:
        __import__(import_name)
        print(f"✓ {display_name} installed")
    except ImportError:
        print(f"✗ {display_name} missing — run: pip install {import_name}")

# Check Ollama connectivity
try:
    import ollama
    client = ollama.Client()
    models = client.list()
    model_names = [m.model for m in models.models]
    print(f"\n✓ Ollama running with {len(model_names)} model(s): {model_names[:5]}")

    if not any("llama" in m.lower() for m in model_names):
        print("  ⚠️  No Llama model found. Run: ollama pull llama3.1:8b")
except Exception as e:
    print(f"\n✗ Ollama not reachable: {e}")
    print("  Run: ollama serve")

# Check embedding model
try:
    import ollama
    ollama.embed(model="nomic-embed-text", input="test")
    print("✓ Embedding model (nomic-embed-text) available")
except Exception:
    print("✗ Embedding model missing — run: ollama pull nomic-embed-text")

print("\n" + "=" * 50)
print("If all checks pass, you're ready for Module 01!")
print("=" * 50)
