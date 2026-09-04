"""Per-notebook retriever cache.

Building a dense retriever embeds every chunk in the notebook. Doing that on
each question would burn the embedding quota (free tier bills per item and caps
at 1000 items/day) and add seconds of latency to every chat turn. So retrievers
are cached against `(notebook_id, version, mode)` — `Notebook.version` is
bumped whenever the source set changes, which invalidates the entry for free.

The cache is process-local and bounded: notebooks are small, but an API server
that runs for days should not grow a retriever per notebook version forever.
"""

from collections import OrderedDict
from threading import Lock

from citefyme.embeddings import get_embedding_provider
from citefyme.factory import build_retriever
from citefyme.library import Notebook

MAX_ENTRIES = 24

_cache: "OrderedDict[tuple[str, int, str], object]" = OrderedDict()
_lock = Lock()


def get_retriever(nb: Notebook, mode: str = "dense"):
    """Cached retriever for this notebook at its current version.

    `hybrid` reuses the cached dense index rather than embedding the corpus a
    second time — the embedding quota is per item, so embedding twice is what
    actually trips the limit."""
    key = (nb.notebook_id, nb.version, mode)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    if mode == "bm25":
        retriever = build_retriever("bm25", nb.sources)
    elif mode == "dense":
        retriever = build_retriever("dense", nb.sources, get_embedding_provider())
    elif mode == "hybrid":
        dense = get_retriever(nb, "dense")
        retriever = build_retriever("hybrid", nb.sources, dense=dense)
    else:
        raise ValueError(f"unknown retrieval mode: {mode}")

    with _lock:
        _cache[key] = retriever
        _cache.move_to_end(key)
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)
    return retriever


def invalidate(notebook_id: str) -> None:
    with _lock:
        for key in [k for k in _cache if k[0] == notebook_id]:
            del _cache[key]
