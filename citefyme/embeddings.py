import os
from abc import ABC, abstractmethod

import numpy as np

from citefyme.observability import Trace
from citefyme.rate_limit import call_with_backoff, embed_limiter

DEFAULT_GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str], trace: Trace | None = None) -> np.ndarray:
        """Return an (n, dim) float array of embeddings."""


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = DEFAULT_GEMINI_EMBEDDING_MODEL):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def embed(self, texts: list[str], trace: Trace | None = None) -> np.ndarray:
        def call():
            embed_limiter.acquire(len(texts))
            return call_with_backoff(
                lambda: self.client.models.embed_content(model=self.model, contents=texts)
            )

        if trace is None:
            result = call()
        else:
            with trace.span("embed.gemini", model=self.model, n_texts=len(texts)):
                result = call()
        return np.array([e.values for e in result.embeddings], dtype=np.float32)


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic, no-API-key fallback: bag-of-words hashed into a fixed-size
    vector. Good enough to sanity-check the dense-retrieval code path without a key;
    not a substitute for real semantic embeddings."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str], trace: Trace | None = None) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vectors[i, hash(word) % self.dim] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


def get_embedding_provider() -> EmbeddingProvider:
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiEmbeddingProvider()
    return HashEmbeddingProvider()
