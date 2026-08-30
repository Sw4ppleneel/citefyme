import json
import os
import re
import uuid
from pathlib import Path

from citefyme.ingest import build_source
from citefyme.models import Source

STORE_PATH = Path(os.environ.get("CITEFYME_HOME", str(Path.home() / ".citefyme"))) / "notebook.json"


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]
    return slug or "source"


class NotebookStore:
    """A single persisted 'notebook' of sources, like NotebookLM: add sources,
    ask questions grounded only in what's been added. Persisted as JSON so it
    survives restarts."""

    def __init__(self, path: Path = STORE_PATH, seed_if_empty=None):
        self.path = path
        self.version = 0
        self.sources: list[Source] = self._load()
        if not self.sources and seed_if_empty:
            for title, authors, year, text in seed_if_empty():
                self.add(title, authors, year, text)

    def _load(self) -> list[Source]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        return [Source(**s) for s in data]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([s.model_dump() for s in self.sources], indent=2))

    def list(self) -> list[Source]:
        return self.sources

    def add(self, title: str, authors: list[str], year: int, text: str) -> Source:
        source_id = f"{_slugify(title)}_{uuid.uuid4().hex[:6]}"
        source = build_source(source_id, title, authors, year, text)
        self.sources.append(source)
        self.version += 1
        self._save()
        return source

    def remove(self, source_id: str) -> bool:
        before = len(self.sources)
        self.sources = [s for s in self.sources if s.source_id != source_id]
        if len(self.sources) != before:
            self.version += 1
            self._save()
            return True
        return False
