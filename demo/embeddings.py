import os

from langchain_openai import OpenAIEmbeddings

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
  if provider == "openai":
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
      raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")
    return OpenAIEmbeddings(model=model, api_key=api_key)

  raise ValueError(f"Unsupported embedding provider: {provider}")
