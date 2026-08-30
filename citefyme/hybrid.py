from citefyme.dense_retrieval import DenseRetriever
from citefyme.models import Chunk
from citefyme.retrieval import BM25Retriever


class HybridRetriever:
    """Fuses BM25 and dense rankings with Reciprocal Rank Fusion (RRF)."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        bm25_ranked = self.bm25.rank_all(query)
        dense_ranked = self.dense.rank_all(query)

        scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        for ranked in (bm25_ranked, dense_ranked):
            for rank, chunk in enumerate(ranked):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + rank + 1
                )
                chunks_by_id[chunk.chunk_id] = chunk

        ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [chunks_by_id[cid] for cid in ranked_ids[:k]]
