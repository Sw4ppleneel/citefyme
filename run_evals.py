from dotenv import load_dotenv

load_dotenv()

from citefyme.embeddings import get_embedding_provider
from citefyme.eval.baseline_compare import evaluate_baseline, evaluate_rag_condition
from citefyme.eval.citation_eval import evaluate_citations
from citefyme.eval.report import print_table
from citefyme.eval.dataset import GOLD_QUESTIONS, UNANSWERABLE_QUESTIONS
from citefyme.eval.retrieval_eval import evaluate_retriever
from citefyme.factory import build_retriever
from citefyme.llm import get_provider
from citefyme.observability import read_recent_traces
from citefyme.reranker import get_reranker
from sample_data.sources import get_sample_sources


def main():
    sources = get_sample_sources()
    provider = get_provider()
    embedder = get_embedding_provider()
    reranker = get_reranker()

    print(f"llm provider: {type(provider).__name__} | embedding provider: {type(embedder).__name__} "
          f"| reranker: {type(reranker).__name__}\n")

    print("## Retrieval eval (Recall@K, MRR over gold question set)\n")
    retrieval_rows = []
    retrievers = {"bm25": build_retriever("bm25", sources)}
    retrievers["dense"] = build_retriever("dense", sources, embedder)
    # hybrid reuses the dense index instead of re-embedding the corpus
    retrievers["hybrid"] = build_retriever("hybrid", sources, dense=retrievers["dense"])
    for mode, retriever in retrievers.items():
        retrieval_rows.append({"mode": mode, **evaluate_retriever(retriever, GOLD_QUESTIONS)})

    hybrid_reranked_metrics = evaluate_retriever(
        retrievers["hybrid"], GOLD_QUESTIONS, reranker=reranker
    )
    retrieval_rows.append({"mode": "hybrid+rerank", **hybrid_reranked_metrics})

    print_table(retrieval_rows, ["mode", "recall@1", "recall@3", "recall@5", "mrr"])

    print("\n## Citation eval (claim-level; bm25 vs hybrid vs hybrid+rerank, to keep free-tier "
          "call volume sane — dense showed near-identical retrieval to bm25 above so it's "
          "skipped here)\n")
    citation_rows = []
    for mode in ("bm25", "hybrid"):
        metrics = evaluate_citations(GOLD_QUESTIONS, sources, provider, retrievers[mode])
        citation_rows.append({"mode": mode, **metrics})
    citation_rows.append({
        "mode": "hybrid+rerank",
        **evaluate_citations(GOLD_QUESTIONS, sources, provider, retrievers["hybrid"], reranker=reranker),
    })
    print_table(
        citation_rows,
        ["mode", "citation_precision", "citation_completeness", "unsupported_rate"],
    )

    print("\n## Naive LLM (no retrieval) vs RAG — same 3 metrics, includes 3 unanswerable "
          "questions the corpus can't actually answer\n")
    compare_rows = [
        {"condition": "naive_llm (no retrieval)",
         **evaluate_baseline(provider, GOLD_QUESTIONS, UNANSWERABLE_QUESTIONS)},
        {"condition": "hybrid_rag",
         **evaluate_rag_condition(GOLD_QUESTIONS, UNANSWERABLE_QUESTIONS, sources, provider, retrievers["hybrid"])},
        {"condition": "hybrid_rag+rerank",
         **evaluate_rag_condition(GOLD_QUESTIONS, UNANSWERABLE_QUESTIONS, sources, provider,
                                   retrievers["hybrid"], reranker=reranker)},
    ]
    print_table(
        compare_rows,
        ["condition", "keyword_recall", "citation_availability", "hallucination_rate"],
    )
    print("\n(keyword_recall: does the answer mention the expected key terms, over the 8 "
          "answerable questions. citation_availability: fraction of claims backed by a real, "
          "checkable source — naive_llm can never have this by construction. hallucination_rate: "
          "fraction of the 3 unanswerable questions where the system asserted a confident, "
          "'supported' answer instead of correctly having no evidence / abstaining.)")

    print("\n## Observability (from this run's traces)\n")
    recent = read_recent_traces(limit=500)
    total_tokens = sum(t["total_tokens"] for t in recent)
    total_llm_calls = sum(t["llm_calls"] for t in recent)
    avg_ms = sum(t["total_ms"] for t in recent) / len(recent) if recent else 0
    print(f"investigations traced: {len(recent)} | total tokens: {total_tokens} | "
          f"total llm calls: {total_llm_calls} | avg investigation latency: {avg_ms:.0f}ms")
    print("(raw traces persisted at ~/.citefyme/traces.jsonl)")


if __name__ == "__main__":
    main()
