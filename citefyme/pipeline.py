from typing import Protocol

from citefyme.llm import LLMProvider
from citefyme.models import Chunk, Claim, CitationState, Evidence, Source
from citefyme.observability import Trace
from citefyme.reranker import Reranker


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Chunk]: ...


def run_investigation(
    question: str,
    sources: list[Source],
    provider: LLMProvider,
    retriever: Retriever,
    k: int = 5,
    reranker: Reranker | None = None,
    rerank_pool: int = 15,
    trace: Trace | None = None,
) -> list[Claim]:
    trace = trace or Trace(name="investigation")
    chunk_lookup = {c.chunk_id: c for s in sources for c in s.chunks}

    with trace.span("retrieve", k=k, pool=rerank_pool if reranker else k) as span:
        candidates = retriever.retrieve(question, k=rerank_pool if reranker else k)
        span.set(n_candidates=len(candidates))

    if reranker is not None:
        with trace.span("rerank", k=k, n_candidates=len(candidates)):
            retrieved = reranker.rerank(question, candidates, top_k=k, trace=trace)
    else:
        retrieved = candidates[:k]

    with trace.span("extract_claims", n_chunks=len(retrieved)) as span:
        raw_claims = provider.extract_claims(question, retrieved, trace=trace)
        span.set(n_claims=len(raw_claims))

    claims = []
    with trace.span("verify_claims", n_claims=len(raw_claims)):
        for i, rc in enumerate(raw_claims):
            evidence = []
            for chunk_id in rc["chunk_ids"]:
                chunk = chunk_lookup.get(chunk_id)
                if chunk:
                    evidence.append(
                        Evidence(
                            evidence_id=f"ev_{i}_{chunk_id}",
                            chunk_id=chunk.chunk_id,
                            source_id=chunk.source_id,
                            section=chunk.section,
                            text=chunk.text,
                        )
                    )

            claim = Claim(claim_id=f"claim_{i}", text=rc["text"], evidence=evidence)

            if evidence:
                state, note = provider.verify_claim(claim.text, evidence[0].text, trace=trace)
                claim.citation_state = CitationState(state)
                claim.verification_note = note
            else:
                claim.citation_state = CitationState.UNSUPPORTED
                claim.verification_note = "no evidence retrieved"

            claims.append(claim)

    trace.persist()
    return claims


STATE_ICON = {
    CitationState.SUPPORTED: "\U0001f7e2",
    CitationState.PARTIAL: "\U0001f7e1",
    CitationState.UNSUPPORTED: "\U0001f534",
    CitationState.CONFLICTING: "⚫",
}


def print_report(question: str, claims: list[Claim]) -> None:
    print(f"# Investigation: {question}\n")
    for claim in claims:
        icon = STATE_ICON[claim.citation_state]
        print(f"{icon} [{claim.citation_state.value}] {claim.text}")
        for ev in claim.evidence:
            print(f"    -> {ev.source_id} / {ev.section}: \"{ev.text[:80]}...\"")
        if claim.verification_note:
            print(f"    note: {claim.verification_note}")
        print()

    total = len(claims)
    supported = sum(1 for c in claims if c.citation_state == CitationState.SUPPORTED)
    print(f"Claims: {total} | Supported: {supported} | "
          f"Citation precision: {supported / total:.0%}" if total else "No claims.")
