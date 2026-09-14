import os
from abc import ABC, abstractmethod

import numpy as np

from citefyme.observability import Trace
from citefyme.rate_limit import call_with_backoff, embed_limiter

DEFAULT_GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
# BatchEmbedContentsRequest hard-rejects over 100 items in one call; matches
# embed_limiter's per-minute headroom so a single batch never straddles a
# throttle wait.
_MAX_BATCH_ITEMS = 90


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
        def call_batch(batch: list[str]):
            embed_limiter.acquire(len(batch))
            return call_with_backoff(
                lambda: self.client.models.embed_content(model=self.model, contents=batch)
            )

        batches = [
            texts[i : i + _MAX_BATCH_ITEMS] for i in range(0, len(texts), _MAX_BATCH_ITEMS)
        ]
        embeddings = []
        if trace is None:
            for batch in batches:
                embeddings.extend(call_batch(batch).embeddings)
        else:
            with trace.span("embed.gemini", model=self.model, n_texts=len(texts)):
                for batch in batches:
                    embeddings.extend(call_batch(batch).embeddings)
        return np.array([e.values for e in embeddings], dtype=np.float32)


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
