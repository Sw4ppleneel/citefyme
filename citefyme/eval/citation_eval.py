from citefyme.eval.dataset import EvalQuestion
from citefyme.llm import LLMProvider
from citefyme.models import CitationState, Source
from citefyme.pipeline import Retriever, run_investigation
from citefyme.reranker import Reranker


def evaluate_citations(
    questions: list[EvalQuestion],
    sources: list[Source],
    provider: LLMProvider,
    retriever: Retriever,
    k: int = 5,
    reranker: Reranker | None = None,
) -> dict[str, float]:
    total_claims = 0
    supported = 0
    unsupported = 0
    claims_with_evidence = 0

    for q in questions:
        claims = run_investigation(
            q.question, sources, provider, retriever, k=k, reranker=reranker
        )
        total_claims += len(claims)
        for c in claims:
            if c.evidence:
                claims_with_evidence += 1
            if c.citation_state == CitationState.SUPPORTED:
                supported += 1
            elif c.citation_state == CitationState.UNSUPPORTED:
                unsupported += 1

    if total_claims == 0:
        return {"citation_precision": 0.0, "citation_completeness": 0.0, "unsupported_rate": 0.0}

    return {
        "citation_precision": supported / total_claims,
        "citation_completeness": claims_with_evidence / total_claims,
        "unsupported_rate": unsupported / total_claims,
    }
