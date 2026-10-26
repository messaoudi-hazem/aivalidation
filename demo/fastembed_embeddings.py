from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
  def __init__(self, model: str, batch_size: int = 32):
    self._model = TextEmbedding(model_name=model)
    self._batch_size = batch_size

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    vectors = self._model.embed(texts, batch_size=self._batch_size)
    return [vector.tolist() for vector in vectors]

  def embed_query(self, text: str) -> list[float]:
    return next(self._model.embed([text])).tolist()
