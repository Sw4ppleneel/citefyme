import os
from abc import ABC, abstractmethod

from citefyme.llm import DEFAULT_GEMINI_MODEL, _extract_json, gemini_generate
from citefyme.models import Chunk
from citefyme.observability import Trace


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int, trace: Trace | None = None) -> list[Chunk]:
        ...


class GeminiReranker(Reranker):
    """Listwise LLM reranking: one call ranks the whole candidate set by
    relevance, instead of a pointwise call per chunk (cheaper, fewer rate-limit
    hits on the free tier)."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def rerank(
        self, query: str, chunks: list[Chunk], top_k: int = 5, trace: Trace | None = None
    ) -> list[Chunk]:
        if not chunks:
            return []

        listing = "\n".join(f"{i}: [{c.section}] {c.text}" for i, c in enumerate(chunks))
        prompt = (
            f"Query: {query}\n\nCandidate passages:\n{listing}\n\n"
            f"Rank all {len(chunks)} passages by relevance to the query, most relevant "
            "first. Respond ONLY with a JSON array of the integer indices in ranked "
            'order, e.g. [3, 0, 2, 1], including every index exactly once.'
        )
        text = gemini_generate(self.client, self.model, prompt, trace, "llm.rerank")
        order = _extract_json(text, "[", "]")

        if not order:
            return chunks[:top_k]

        seen: set[int] = set()
        ranked: list[Chunk] = []
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                ranked.append(chunks[idx])
                seen.add(idx)
        for i, c in enumerate(chunks):
            if i not in seen:
                ranked.append(c)
        return ranked[:top_k]


class NullReranker(Reranker):
    """No-op fallback when no API key is available: preserves incoming order."""

    def rerank(
        self, query: str, chunks: list[Chunk], top_k: int = 5, trace: Trace | None = None
    ) -> list[Chunk]:
        return chunks[:top_k]


def get_reranker() -> Reranker:
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiReranker()
    return NullReranker()
