from rank_bm25 import BM25Okapi

from citefyme.models import Chunk, Source


class BM25Retriever:
    def __init__(self, sources: list[Source]):
        self.chunks: list[Chunk] = [c for s in sources for c in s.chunks]
        tokenized = [c.text.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return self.rank_all(query)[:k]

    def rank_all(self, query: str) -> list[Chunk]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in ranked if score > 0]
