from citefyme.eval.dataset import EvalQuestion
from citefyme.pipeline import Retriever
from citefyme.reranker import Reranker


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    retriever: Retriever,
    questions: list[EvalQuestion],
    k_values: tuple[int, ...] = (1, 3, 5),
    reranker: Reranker | None = None,
    rerank_pool: int = 15,
) -> dict[str, float]:
    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    max_k = max(k_values)

    for q in questions:
        if reranker is not None:
            candidates = retriever.retrieve(q.question, k=rerank_pool)
            retrieved = reranker.rerank(q.question, candidates, top_k=max_k)
        else:
            retrieved = retriever.retrieve(q.question, k=max_k)
        retrieved_ids = [c.chunk_id for c in retrieved]

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(retrieved_ids, q.relevant_chunk_ids, k))
        results["mrr"].append(reciprocal_rank(retrieved_ids, q.relevant_chunk_ids))

    return {metric: sum(vals) / len(vals) for metric, vals in results.items()}
