from citefyme.eval.dataset import EvalQuestion
from citefyme.llm import LLMProvider
from citefyme.models import CitationState, Source
from citefyme.pipeline import Retriever, run_investigation
from citefyme.reranker import Reranker

UNCERTAINTY_PHRASES = [
    "don't know", "do not know", "not sure", "no information", "not enough information",
    "cannot determine", "can't determine", "not confident", "unable to determine",
    "i don't have", "i do not have", "not aware", "cannot verify", "can't verify",
    "no reliable information", "not publicly", "not disclosed", "unclear",
]


def keyword_recall(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    text = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def expresses_uncertainty(answer: str) -> bool:
    text = answer.lower()
    return any(p in text for p in UNCERTAINTY_PHRASES)


def evaluate_baseline(
    provider: LLMProvider, answerable: list[EvalQuestion], unanswerable: list[EvalQuestion]
) -> dict:
    recalls = [
        r
        for q in answerable
        if (r := keyword_recall(provider.answer_directly(q.question), q.expected_keywords)) is not None
    ]

    hallucinations = sum(
        0 if expresses_uncertainty(provider.answer_directly(q.question)) else 1
        for q in unanswerable
    )

    return {
        "keyword_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "citation_availability": 0.0,
        "hallucination_rate": hallucinations / len(unanswerable) if unanswerable else 0.0,
    }


def evaluate_rag_condition(
    answerable: list[EvalQuestion],
    unanswerable: list[EvalQuestion],
    sources: list[Source],
    provider: LLMProvider,
    retriever: Retriever,
    reranker: Reranker | None = None,
    k: int = 5,
) -> dict:
    recalls = []
    claims_with_evidence = 0
    total_claims = 0
    for q in answerable:
        claims = run_investigation(q.question, sources, provider, retriever, k=k, reranker=reranker)
        answer_text = " ".join(c.text for c in claims)
        r = keyword_recall(answer_text, q.expected_keywords)
        if r is not None:
            recalls.append(r)
        total_claims += len(claims)
        claims_with_evidence += sum(1 for c in claims if c.evidence)

    hallucinations = 0
    for q in unanswerable:
        claims = run_investigation(q.question, sources, provider, retriever, k=k, reranker=reranker)
        if any(c.citation_state == CitationState.SUPPORTED for c in claims):
            hallucinations += 1

    return {
        "keyword_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "citation_availability": claims_with_evidence / total_claims if total_claims else 0.0,
        "hallucination_rate": hallucinations / len(unanswerable) if unanswerable else 0.0,
    }
