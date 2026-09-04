"""Retrieval-latency benchmark: same corpus, same gold questions, same modes as
run_topic_eval.py's Recall@K/MRR section — this measures wall-clock cost
instead of quality, because a mode that scores best on paper still has to be
worth waiting for.

    .venv/bin/python run_latency_bench.py rag-methods

Only `retrieve()` (or `retrieve()` + `rerank()`) is timed — index *build*
(embedding the whole corpus once) is reported separately since it happens
once per notebook version, not once per question.
"""

import argparse
import statistics
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from citefyme.corpora import get_corpus
from citefyme.embeddings import get_embedding_provider
from citefyme.eval.report import print_table
from citefyme.eval.synthetic import get_goldset
from citefyme.factory import build_retriever
from citefyme.llm import get_provider
from citefyme.reranker import get_reranker
from citefyme.topics import TOPICS


def timed(fn) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn()
    return (time.perf_counter() - start) * 1000, result


def bench_mode(name, retrieve_fn, questions, k=5):
    latencies = []
    for q in questions:
        ms, _ = timed(lambda: retrieve_fn(q.question, k))
        latencies.append(ms)
    latencies.sort()
    n = len(latencies)
    return {
        "mode": name,
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[min(n - 1, int(n * 0.95))],
        "mean_ms": statistics.mean(latencies),
        "max_ms": max(latencies),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", choices=sorted(TOPICS))
    ap.add_argument("--questions", type=int, default=8)
    args = ap.parse_args()

    topic = TOPICS[args.topic]
    provider = get_provider()
    embedder = get_embedding_provider()
    reranker = get_reranker()
    sources = get_corpus(topic)
    answerable, unanswerable = get_goldset(topic.slug, sources, provider, n_answerable=args.questions)
    questions = answerable + unanswerable
    print(f"# {topic.label}: retrieval latency ({len(sources)} papers, "
          f"{sum(len(s.chunks) for s in sources)} chunks, {len(questions)} queries)\n")

    build_ms, bm25 = timed(lambda: build_retriever("bm25", sources))
    print(f"index build  bm25:   {build_ms:8.0f}ms  (in-memory, no network call)")
    build_ms, dense = timed(lambda: build_retriever("dense", sources, embedder))
    print(f"index build  dense:  {build_ms:8.0f}ms  ({sum(len(s.chunks) for s in sources)} chunks "
          f"embedded in one batched API call)")
    hybrid = build_retriever("hybrid", sources, dense=dense)
    print()

    rows = [
        bench_mode("bm25", lambda q, k: bm25.retrieve(q, k), questions),
        bench_mode("dense", lambda q, k: dense.retrieve(q, k), questions),
        bench_mode("hybrid", lambda q, k: hybrid.retrieve(q, k), questions),
        bench_mode(
            "hybrid+rerank",
            lambda q, k: reranker.rerank(q, hybrid.retrieve(q, 15), top_k=k),
            questions,
        ),
    ]
    print_table(rows, ["mode", "p50_ms", "p95_ms", "mean_ms", "max_ms"])
    print("\nEach row is one retrieve() call per gold question (k=5; hybrid+rerank pools 15 "
          "candidates before reranking) — no claim extraction or verification, so this isolates "
          "retrieval cost from the LLM cost already measured in run_topic_eval.py's citation section.")


if __name__ == "__main__":
    sys.exit(main())
