import json
import os
import re
from abc import ABC, abstractmethod

from citefyme.models import Chunk
from citefyme.observability import Trace
from citefyme.rate_limit import call_with_backoff, generate_limiter

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def _extract_json(raw: str, opener: str, closer: str):
    """Parse the first complete JSON value starting at `opener`, ignoring any
    trailing prose the model appended after it. A naive rfind(closer) slice
    breaks whenever that trailing text itself contains a stray closer
    character (observed in real Gemini responses)."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    start = cleaned.find(opener)
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return obj
    except json.JSONDecodeError:
        return None


def _call_generate(client, model: str, prompt: str):
    generate_limiter.acquire()
    return call_with_backoff(lambda: client.models.generate_content(model=model, contents=prompt))


def gemini_generate(client, model: str, prompt: str, trace: Trace | None, span_name: str) -> str:
    if trace is None:
        return _call_generate(client, model, prompt).text
    with trace.span(span_name, model=model) as span:
        resp = _call_generate(client, model, prompt)
        usage = resp.usage_metadata
        span.set(
            prompt_tokens=usage.prompt_token_count,
            completion_tokens=usage.candidates_token_count,
            total_tokens=usage.total_token_count,
        )
        return resp.text


class LLMProvider(ABC):
    @abstractmethod
    def extract_claims(
        self, question: str, chunks: list[Chunk], trace: Trace | None = None
    ) -> list[dict]:
        """Return [{"text": ..., "chunk_ids": [...]}]"""

    @abstractmethod
    def verify_claim(
        self, claim_text: str, evidence_text: str, trace: Trace | None = None
    ) -> tuple[str, str]:
        """Return (citation_state, note)"""

    @abstractmethod
    def answer_directly(self, question: str, trace: Trace | None = None) -> str:
        """Naive no-retrieval baseline: answer from parametric knowledge alone."""

    @abstractmethod
    def complete(self, prompt: str, trace: Trace | None = None) -> str:
        """Raw single-prompt completion, no wrapper instructions. Used by tooling
        around the pipeline (eval-set generation), never by the pipeline itself."""


_DIRECT_ANSWER_PROMPT = (
    "Answer this question directly and concisely (2-4 sentences), using only your "
    "own knowledge. If you are not confident or do not have reliable information, "
    "say so explicitly instead of guessing.\n\nQuestion: {question}"
)


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-5"):
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model

    def extract_claims(
        self, question: str, chunks: list[Chunk], trace: Trace | None = None
    ) -> list[dict]:
        context = "\n\n".join(f"[{c.chunk_id}] ({c.section}): {c.text}" for c in chunks)
        prompt = (
            f"Question: {question}\n\nEvidence chunks:\n{context}\n\n"
            "Extract substantive claims that answer the question, using ONLY the "
            "evidence above. For each claim, cite the exact chunk_id(s) that support it. "
            "Respond ONLY with a JSON array: "
            '[{"text": "...", "chunk_ids": ["chunk_id", ...]}]'
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_json(resp.content[0].text, "[", "]") or []

    def verify_claim(
        self, claim_text: str, evidence_text: str, trace: Trace | None = None
    ) -> tuple[str, str]:
        prompt = (
            f'Claim: "{claim_text}"\nEvidence: "{evidence_text}"\n\n'
            "Does the evidence support the claim exactly as stated? "
            'Respond ONLY with JSON: {"state": "supported|partial|unsupported", "note": "..."}'
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = _extract_json(resp.content[0].text, "{", "}")
        if not parsed:
            return "unsupported", "could not parse verification response"
        return parsed.get("state", "unsupported"), parsed.get("note", "")

    def answer_directly(self, question: str, trace: Trace | None = None) -> str:
        return self.complete(_DIRECT_ANSWER_PROMPT.format(question=question), trace)

    def complete(self, prompt: str, trace: Trace | None = None) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_GEMINI_MODEL):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def _generate(self, prompt: str, trace: Trace | None, span_name: str) -> str:
        return gemini_generate(self.client, self.model, prompt, trace, span_name)

    def extract_claims(
        self, question: str, chunks: list[Chunk], trace: Trace | None = None
    ) -> list[dict]:
        context = "\n\n".join(f"[{c.chunk_id}] ({c.section}): {c.text}" for c in chunks)
        prompt = (
            f"Question: {question}\n\nEvidence chunks:\n{context}\n\n"
            "Extract substantive claims that answer the question, using ONLY the "
            "evidence above. For each claim, cite the exact chunk_id(s) that support it. "
            "Respond ONLY with a JSON array, no markdown fences: "
            '[{"text": "...", "chunk_ids": ["chunk_id", ...]}]'
        )
        text = self._generate(prompt, trace, "llm.extract_claims")
        return _extract_json(text, "[", "]") or []

    def verify_claim(
        self, claim_text: str, evidence_text: str, trace: Trace | None = None
    ) -> tuple[str, str]:
        prompt = (
            f'Claim: "{claim_text}"\nEvidence: "{evidence_text}"\n\n'
            "Does the evidence support the claim exactly as stated? "
            "Respond ONLY with JSON, no markdown fences: "
            '{"state": "supported|partial|unsupported", "note": "..."}'
        )
        text = self._generate(prompt, trace, "llm.verify_claim")
        parsed = _extract_json(text, "{", "}")
        if not parsed:
            return "unsupported", "could not parse verification response"
        return parsed.get("state", "unsupported"), parsed.get("note", "")

    def answer_directly(self, question: str, trace: Trace | None = None) -> str:
        return self._generate(
            _DIRECT_ANSWER_PROMPT.format(question=question), trace, "llm.answer_directly"
        )

    def complete(self, prompt: str, trace: Trace | None = None) -> str:
        return self._generate(prompt, trace, "llm.complete")


class StubProvider(LLMProvider):
    """Deterministic, no-API-key fallback: one claim per retrieved chunk,
    verification via lexical overlap. Swap for GeminiProvider once a key is set."""

    def extract_claims(
        self, question: str, chunks: list[Chunk], trace: Trace | None = None
    ) -> list[dict]:
        claims = []
        for c in chunks:
            sentence = c.text.strip().split(".")[0].strip() + "."
            claims.append({"text": sentence, "chunk_ids": [c.chunk_id]})
        return claims

    def verify_claim(
        self, claim_text: str, evidence_text: str, trace: Trace | None = None
    ) -> tuple[str, str]:
        claim_words = set(w.lower() for w in re.findall(r"\w+", claim_text))
        evidence_words = set(w.lower() for w in re.findall(r"\w+", evidence_text))
        overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
        if overlap > 0.6:
            return "supported", f"lexical overlap {overlap:.0%}"
        elif overlap > 0.3:
            return "partial", f"lexical overlap {overlap:.0%}"
        return "unsupported", f"lexical overlap {overlap:.0%}"

    def answer_directly(self, question: str, trace: Trace | None = None) -> str:
        return "[stub] no parametric knowledge available without a real LLM key."

    def complete(self, prompt: str, trace: Trace | None = None) -> str:
        return ""


def get_provider() -> LLMProvider:
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    return StubProvider()
