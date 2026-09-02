"""Fetch-once / reuse-forever corpora on disk.

A corpus is just a list of `Source`, so it round-trips through the same
pydantic models the rest of the pipeline uses. Caching matters for evals:
re-fetching arXiv between runs would silently change the denominator.
"""

import json
import os
from pathlib import Path

from citefyme.arxiv import fetch_corpus_multi
from citefyme.models import Source
from citefyme.topics import TOPICS, Topic

CORPUS_DIR = Path(os.environ.get("CITEFYME_HOME", str(Path.home() / ".citefyme"))) / "corpora"


def corpus_path(slug: str) -> Path:
    return CORPUS_DIR / f"{slug}.json"


def load_corpus(slug: str) -> list[Source] | None:
    path = corpus_path(slug)
    if not path.exists():
        return None
    return [Source(**s) for s in json.loads(path.read_text())]


def save_corpus(slug: str, sources: list[Source]) -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path(slug).write_text(json.dumps([s.model_dump() for s in sources], indent=2))


def get_corpus(topic: Topic | str, refresh: bool = False) -> list[Source]:
    topic = TOPICS[topic] if isinstance(topic, str) else topic
    if not refresh:
        cached = load_corpus(topic.slug)
        if cached:
            return cached
    sources = fetch_corpus_multi(
        topic.queries,
        per_query=topic.per_query,
        exclude_terms=topic.exclude_terms,
        limit=topic.limit,
    )
    save_corpus(topic.slug, sources)
    return sources
