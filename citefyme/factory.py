from citefyme.dense_retrieval import DenseRetriever
from citefyme.embeddings import EmbeddingProvider
from citefyme.hybrid import HybridRetriever
from citefyme.models import Source
from citefyme.retrieval import BM25Retriever

RETRIEVAL_MODES = ("bm25", "dense", "hybrid")


def build_retriever(
    mode: str,
    sources: list[Source],
    embedder: EmbeddingProvider | None = None,
    dense: DenseRetriever | None = None,
):
    """`dense` lets a caller that already built a DenseRetriever hand it to the
    hybrid one instead of re-embedding the whole corpus — the embedding quota
    is per item, so embedding twice is what actually trips the rate limit."""
    if mode == "bm25":
        return BM25Retriever(sources)
    if mode == "dense":
        if dense is not None:
            return dense
        if embedder is None:
            raise ValueError("dense retrieval requires an embedder")
        return DenseRetriever(sources, embedder)
    if mode == "hybrid":
        if dense is None:
            if embedder is None:
                raise ValueError("hybrid retrieval requires an embedder")
            dense = DenseRetriever(sources, embedder)
        return HybridRetriever(BM25Retriever(sources), dense)
    raise ValueError(f"unknown retrieval mode: {mode} (choices: {RETRIEVAL_MODES})")
