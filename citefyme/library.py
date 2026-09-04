"""Multi-notebook persistence.

`store.NotebookStore` holds exactly one notebook, which was enough for the CLI
demo. The UI needs many — one per research topic — so this module owns a
library of notebooks, each with:

  * `sources`     — confirmed papers, already chunked, what retrieval runs over
  * `candidates`  — papers a search surfaced, awaiting confirm/reject. Kept
                    after the decision so a rejected paper does not resurface
                    on the next search, and so the review trail is visible.
  * `chat`        — the question/claims history, so a notebook reads like a
                    conversation rather than a one-shot query box.

One JSON file per notebook under `$CITEFYME_HOME/notebooks/`, plus nothing
else: the library index is derived by reading the directory, so there is no
index file that can drift out of sync with the notebooks themselves.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from citefyme.arxiv import paper_to_source
from citefyme.models import Claim, Source

HOME = Path(os.environ.get("CITEFYME_HOME", str(Path.home() / ".citefyme")))
NOTEBOOKS_DIR = HOME / "notebooks"


def _slug(text: str, n: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n] or "paper"


def _now() -> float:
    return time.time()


class Candidate(BaseModel):
    """A paper a search turned up, plus the decision made about it."""

    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    abstract: str = ""
    url: str = ""
    query: str = ""
    status: str = "pending"  # pending | confirmed | rejected
    source_id: str | None = None  # set once confirmed, joins to Source
    decided_at: float | None = None


class ChatTurn(BaseModel):
    turn_id: str
    question: str
    claims: list[Claim] = Field(default_factory=list)
    mode: str = "dense"
    rerank: bool = False
    latency_ms: float = 0.0
    trace_id: str | None = None
    created_at: float = Field(default_factory=_now)
    error: str | None = None


class Notebook(BaseModel):
    notebook_id: str
    title: str
    emoji: str = "\U0001f4d2"
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)
    sources: list[Source] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    chat: list[ChatTurn] = Field(default_factory=list)
    # bumped on every change to the source set; the retriever cache keys on it
    version: int = 0

    @property
    def chunk_count(self) -> int:
        return sum(len(s.chunks) for s in self.sources)

    def summary(self) -> dict:
        return {
            "notebook_id": self.notebook_id,
            "title": self.title,
            "emoji": self.emoji,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_count": len(self.sources),
            "chunk_count": self.chunk_count,
            "pending_count": sum(1 for c in self.candidates if c.status == "pending"),
            "turn_count": len(self.chat),
        }


class Library:
    def __init__(self, root: Path = NOTEBOOKS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, notebook_id: str) -> Path:
        return self.root / f"{notebook_id}.json"

    # --- notebooks -------------------------------------------------------

    def list(self) -> list[Notebook]:
        books = []
        for p in self.root.glob("*.json"):
            try:
                books.append(Notebook(**json.loads(p.read_text())))
            except Exception:
                # a half-written or hand-edited file should not take the whole
                # library down — skip it and keep listing the rest
                continue
        return sorted(books, key=lambda b: b.updated_at, reverse=True)

    def get(self, notebook_id: str) -> Notebook | None:
        p = self._path(notebook_id)
        if not p.exists():
            return None
        return Notebook(**json.loads(p.read_text()))

    def save(self, nb: Notebook) -> Notebook:
        nb.updated_at = _now()
        self._path(nb.notebook_id).write_text(nb.model_dump_json(indent=2))
        return nb

    def create(self, title: str, emoji: str = "\U0001f4d2") -> Notebook:
        nb = Notebook(
            notebook_id=f"{_slug(title, 24)}_{uuid.uuid4().hex[:6]}",
            title=title or "Untitled notebook",
            emoji=emoji,
        )
        return self.save(nb)

    def delete(self, notebook_id: str) -> bool:
        p = self._path(notebook_id)
        if not p.exists():
            return False
        p.unlink()
        return True

    # --- candidates ------------------------------------------------------

    def add_candidates(self, nb: Notebook, papers: list[dict], query: str) -> list[Candidate]:
        """Merge search hits in, skipping any paper already decided on.

        Returns the candidates that correspond to this result set (including
        already-decided ones, so the UI can show 'already in this notebook'
        rather than silently dropping a hit)."""
        by_id = {c.arxiv_id: c for c in nb.candidates}
        out = []
        for p in papers:
            base = p["arxiv_id"].split("v")[0]
            existing = by_id.get(base)
            if existing:
                out.append(existing)
                continue
            cand = Candidate(
                arxiv_id=base,
                title=p["title"],
                authors=p.get("authors", []),
                year=p.get("year", 0),
                abstract=p.get("abstract", ""),
                url=p.get("url", ""),
                query=query,
            )
            nb.candidates.append(cand)
            by_id[base] = cand
            out.append(cand)
        self.save(nb)
        return out

    def decide(self, nb: Notebook, arxiv_ids: list[str], status: str) -> Notebook:
        """Confirm or reject candidates. Confirming chunks the abstract into a
        Source; rejecting drops any Source that was previously built from it."""
        wanted = {a.split("v")[0] for a in arxiv_ids}
        existing_source_ids = {s.source_id for s in nb.sources}
        changed = False

        for cand in nb.candidates:
            if cand.arxiv_id not in wanted or cand.status == status:
                continue
            cand.status = status
            cand.decided_at = _now()

            if status == "confirmed":
                source_id = cand.source_id or f"arxiv_{_slug(cand.title, 32)}"
                if source_id in existing_source_ids:
                    source_id = f"{source_id}_{cand.arxiv_id.replace('.', '_')}"
                cand.source_id = source_id
                nb.sources.append(
                    paper_to_source(
                        {
                            "title": cand.title,
                            "authors": cand.authors,
                            "year": cand.year,
                            "abstract": cand.abstract,
                        },
                        source_id,
                    )
                )
                existing_source_ids.add(source_id)
                changed = True
            elif cand.source_id:
                nb.sources = [s for s in nb.sources if s.source_id != cand.source_id]
                existing_source_ids.discard(cand.source_id)
                cand.source_id = None
                changed = True

        if changed:
            nb.version += 1
        self.save(nb)
        return nb

    def remove_source(self, nb: Notebook, source_id: str) -> Notebook:
        before = len(nb.sources)
        nb.sources = [s for s in nb.sources if s.source_id != source_id]
        if len(nb.sources) != before:
            nb.version += 1
            for c in nb.candidates:
                if c.source_id == source_id:
                    c.status = "rejected"
                    c.source_id = None
                    c.decided_at = _now()
        self.save(nb)
        return nb

    # --- chat ------------------------------------------------------------

    def add_turn(self, nb: Notebook, turn: ChatTurn) -> Notebook:
        nb.chat.append(turn)
        self.save(nb)
        return nb
