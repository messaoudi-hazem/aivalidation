from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


class BatchedOllamaEmbeddings(Embeddings):
  def __init__(self, model: str, base_url: str = "http://localhost:11434", batch_size: int = 16):
    self._embeddings = OllamaEmbeddings(model=model, base_url=base_url)
    self._batch_size = batch_size

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), self._batch_size):
      vectors.extend(self._embeddings.embed_documents(texts[start:start + self._batch_size]))
    return vectors

  def embed_query(self, text: str) -> list[float]:
    return self._embeddings.embed_query(text)
