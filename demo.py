import sys

from dotenv import load_dotenv

load_dotenv()

from citefyme.embeddings import get_embedding_provider
from citefyme.factory import build_retriever
from citefyme.llm import get_provider
from citefyme.observability import Trace
from citefyme.pipeline import print_report, run_investigation
from citefyme.reranker import get_reranker
from sample_data.sources import get_sample_sources


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    use_rerank = "--rerank" in sys.argv
    question = "How do world models learn long-horizon dynamics and what are their failure modes?"

    sources = get_sample_sources()
    provider = get_provider()
    embedder = get_embedding_provider() if mode in ("dense", "hybrid") else None
    retriever = build_retriever(mode, sources, embedder)
    reranker = get_reranker() if use_rerank else None

    print(f"(retrieval: {mode}{'+rerank' if use_rerank else ''} | llm: {type(provider).__name__}"
          + (f" | embeddings: {type(embedder).__name__}" if embedder else "") + ")\n")

    trace = Trace(name=f"demo:{mode}")
    claims = run_investigation(question, sources, provider, retriever, reranker=reranker, trace=trace)
    print_report(question, claims)

    print(f"\n[trace {trace.trace_id}] {trace.total_ms():.0f}ms total | "
          f"{trace.llm_call_count()} llm calls | {trace.total_tokens()} tokens")
    for span in trace.spans:
        print(f"  - {span.name}: {span.to_dict()['duration_ms']}ms  {span.metadata}")


if __name__ == "__main__":
    main()
