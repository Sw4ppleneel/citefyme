"""Supplement to run_topic_eval.py: answer-quality (keyword_recall /
citation_availability / hallucination_rate) for the retrieval modes the stock
baseline section skips — bm25-only and dense-only RAG.

    .venv/bin/python run_rag_variants.py rag-methods
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from citefyme.corpora import get_corpus
from citefyme.embeddings import get_embedding_provider
from citefyme.eval.baseline_compare import evaluate_rag_condition
from citefyme.eval.report import print_table
from citefyme.eval.synthetic import get_goldset
from citefyme.factory import build_retriever
from citefyme.llm import get_provider
from citefyme.topics import TOPICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", choices=sorted(TOPICS))
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--modes", default="bm25,dense")
    args = ap.parse_args()

    topic = TOPICS[args.topic]
    provider = get_provider()
    embedder = get_embedding_provider()
    sources = get_corpus(topic)
    answerable, unanswerable = get_goldset(
        topic.slug, sources, provider, n_answerable=args.questions
    )
    print(f"# {topic.label}: RAG answer quality by retrieval mode")
    print(f"llm: {type(provider).__name__}({getattr(provider,'model','n/a')}) | "
          f"{len(answerable)} answerable + {len(unanswerable)} unanswerable\n")

    rows = []
    for mode in [m.strip() for m in args.modes.split(",")]:
        r = build_retriever(mode, sources, embedder) if mode != "bm25" \
            else build_retriever("bm25", sources)
        rows.append({"condition": f"{mode}_rag",
                     **evaluate_rag_condition(answerable, unanswerable, sources, provider, r)})
    print_table(rows, ["condition", "keyword_recall", "citation_availability",
                       "hallucination_rate"])


if __name__ == "__main__":
    sys.exit(main())
