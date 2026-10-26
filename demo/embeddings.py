import os

try:
  from .ollama_embeddings import BatchedOllamaEmbeddings
except ImportError:
  from ollama_embeddings import BatchedOllamaEmbeddings


def create_embeddings():
  provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
  model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

  if provider == "ollama":
    return BatchedOllamaEmbeddings(
      model=model,
      base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
  if provider == "fastembed":
    try:
      from .fastembed_embeddings import FastEmbedEmbeddings
    except ImportError:
      from fastembed_embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model=model)

  raise ValueError(f"Unsupported embedding provider: {provider}")
