"""Run the full CitefyMe eval battery against a real arXiv corpus.

    .venv/bin/python run_topic_eval.py tiny-world-models
    .venv/bin/python run_topic_eval.py rag-methods --refresh-corpus

Corpus and gold set are cached under ~/.citefyme/ after the first run, so
repeat runs measure the same thing.
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from citefyme.corpora import get_corpus
from citefyme.embeddings import get_embedding_provider
from citefyme.eval.baseline_compare import evaluate_baseline, evaluate_rag_condition
from citefyme.eval.citation_eval import evaluate_citations
from citefyme.eval.report import print_table
from citefyme.eval.retrieval_eval import evaluate_retriever
from citefyme.eval.synthetic import generate_goldset, get_goldset
from citefyme.factory import build_retriever
from citefyme.llm import get_provider
from citefyme.observability import Trace, read_traces_since, trace_count
from citefyme.pipeline import print_report, run_investigation
from citefyme.reranker import get_reranker
from citefyme.topics import TOPICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", choices=sorted(TOPICS))
    ap.add_argument("--refresh-corpus", action="store_true")
    ap.add_argument("--refresh-goldset", action="store_true")
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--skip-eval", action="store_true",
                    help="fetch corpus + run the headline investigation only")
    ap.add_argument("--sections", default="headline,retrieval,citation,baseline",
                    help="comma-separated subset to run, so a section lost to a quota "
                         "cap can be re-run on its own")
    ap.add_argument("--modes", default="bm25,dense,hybrid",
                    help="comma-separated retrieval modes to build — drop dense/hybrid "
                         "to run entirely off the embed quota when it's exhausted "
                         "(bm25 needs no API calls at all)")
    args = ap.parse_args()
    sections = {s.strip() for s in args.sections.split(",")}

    topic = TOPICS[args.topic]
    provider = get_provider()
    embedder = get_embedding_provider()
    reranker = get_reranker()

    print(f"# Topic: {topic.label}  ({topic.slug})")
    print(f"llm: {type(provider).__name__}({getattr(provider, 'model', 'n/a')}) | "
          f"embeddings: {type(embedder).__name__}({getattr(embedder, 'model', 'n/a')}) | "
          f"reranker: {type(reranker).__name__} | sections: {sorted(sections)}")
    if topic.notes:
        print(f"note: {topic.notes}")

    sources = get_corpus(topic, refresh=args.refresh_corpus)
    n_chunks = sum(len(s.chunks) for s in sources)
    kind = "full text" if topic.fulltext else "abstracts"
    print(f"\n## Corpus: {len(sources)} arXiv papers, {n_chunks} chunks "
          f"({kind}, {min(s.year for s in sources)}-{max(s.year for s in sources)})\n")
    for s in sources:
        print(f"  {s.year}  {s.source_id[:40]:42} {len(s.chunks)}c  {s.title[:66]}")

    trace_offset = trace_count()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    retrievers = {}
    if "bm25" in modes:
        retrievers["bm25"] = build_retriever("bm25", sources)
    if "dense" in modes:
        retrievers["dense"] = build_retriever("dense", sources, embedder)
    if "hybrid" in modes:
        # reuse an already-embedded dense retriever rather than paying for it twice
        retrievers["hybrid"] = build_retriever("hybrid", sources, embedder, dense=retrievers.get("dense"))
    if not retrievers:
        raise ValueError(f"--modes matched nothing (got {args.modes!r})")
    # richest available mode drives the headline investigation and the single
    # "RAG" condition in the baseline section
    primary_mode = next(m for m in ("hybrid", "dense", "bm25") if m in retrievers)
    primary = retrievers[primary_mode]

    if "headline" in sections:
        print(f"\n## Headline investigation  (retriever: {primary_mode})\n")
        trace = Trace(name=f"topic:{topic.slug}")
        claims = run_investigation(
            topic.question, sources, provider, primary, reranker=reranker, trace=trace
        )
        print_report(topic.question, claims)
        print(f"\n[trace {trace.trace_id}] {trace.total_ms():.0f}ms | "
              f"{trace.llm_call_count()} llm calls | {trace.total_tokens()} tokens")

    if args.skip_eval:
        return

    answerable, unanswerable = get_goldset(
        topic.slug, sources, provider, n_answerable=args.questions,
        refresh=args.refresh_goldset,
        content_desc="the full text" if topic.fulltext else "the abstracts",
    )
    print(f"\n## Gold set: {len(answerable)} answerable (synthetic, chunk-seeded) + "
          f"{len(unanswerable)} unanswerable\n")
    for q in answerable:
        print(f"  [{q.relevant_chunk_ids[0]}] {q.question}")
        print(f"      keywords: {q.expected_keywords}")
    for q in unanswerable:
        print(f"  [none] {q.question}")

    if "retrieval" in sections:
        print("\n## Retrieval eval (Recall@K, MRR)\n")
        rows = []
        for mode, r in retrievers.items():
            rows.append({"mode": mode, **evaluate_retriever(r, answerable)})
        if "hybrid" in retrievers:
            rows.append({
                "mode": "hybrid+rerank",
                **evaluate_retriever(retrievers["hybrid"], answerable, reranker=reranker),
            })
        print_table(rows, ["mode", "recall@1", "recall@3", "recall@5", "mrr"])

    if "citation" in sections:
        print("\n## Citation eval (claim-level)\n")
        rows = []
        for mode, r in retrievers.items():
            rows.append({"mode": mode,
                         **evaluate_citations(answerable, sources, provider, r)})
        if "hybrid" in retrievers:
            rows.append({"mode": "hybrid+rerank",
                         **evaluate_citations(answerable, sources, provider, retrievers["hybrid"],
                                              reranker=reranker)})
        print_table(rows, ["mode", "citation_precision", "citation_completeness",
                           "unsupported_rate"])

    if "baseline" not in sections:
        return
    print(f"\n## Naive LLM (no retrieval) vs RAG  (retriever: {primary_mode})\n")
    rows = [
        {"condition": "naive_llm (no retrieval)",
         **evaluate_baseline(provider, answerable, unanswerable)},
        {"condition": f"{primary_mode}_rag",
         **evaluate_rag_condition(answerable, unanswerable, sources, provider, primary)},
    ]
    if "hybrid" in retrievers:
        rows.append({"condition": "hybrid_rag+rerank",
                     **evaluate_rag_condition(answerable, unanswerable, sources, provider,
                                              retrievers["hybrid"], reranker=reranker)})
    print_table(rows, ["condition", "keyword_recall", "citation_availability", "hallucination_rate"])

    new_traces = read_traces_since(trace_offset)
    if new_traces:
        print(f"\n## Observability (this run)\n")
        print(f"investigations: {len(new_traces)} | "
              f"llm calls: {sum(t['llm_calls'] for t in new_traces)} | "
              f"tokens: {sum(t['total_tokens'] for t in new_traces)} | "
              f"avg latency: {sum(t['total_ms'] for t in new_traces)/len(new_traces):.0f}ms")


if __name__ == "__main__":
    sys.exit(main())
