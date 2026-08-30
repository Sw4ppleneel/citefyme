import numpy as np

from citefyme.embeddings import EmbeddingProvider
from citefyme.models import Chunk, Source


class DenseRetriever:
    def __init__(self, sources: list[Source], embedder: EmbeddingProvider):
        self.chunks: list[Chunk] = [c for s in sources for c in s.chunks]
        self.embedder = embedder
        self.matrix = embedder.embed([c.text for c in self.chunks])

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return self.rank_all(query)[:k]

    def rank_all(self, query: str) -> list[Chunk]:
        query_vec = self.embedder.embed([query])[0]
        sims = self._cosine_sim(query_vec, self.matrix)
        ranked = sorted(zip(sims, self.chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in ranked]

    @staticmethod
    def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return m_norm @ q_norm
