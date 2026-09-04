"""FastAPI layer over the CitefyMe pipeline, for the notebook UI.

    .venv/bin/uvicorn api.main:app --reload --port 8002

Every route is a thin adapter: the notebook library owns persistence, the
retriever cache owns index lifetime, and `run_investigation` stays the single
path that answers a question. Nothing about the eval harness changes.
"""

import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from citefyme.arxiv import search_arxiv
from citefyme.embeddings import get_embedding_provider
from citefyme.index_cache import get_retriever, invalidate
from citefyme.library import ChatTurn, Library, Notebook
from citefyme.llm import get_provider
from citefyme.observability import Trace
from citefyme.pipeline import run_investigation
from citefyme.reranker import get_reranker

app = FastAPI(title="CitefyMe", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

library = Library()


def _nb(notebook_id: str) -> Notebook:
    nb = library.get(notebook_id)
    if nb is None:
        raise HTTPException(404, f"no notebook {notebook_id}")
    return nb


# --- schemas -------------------------------------------------------------


class CreateNotebook(BaseModel):
    title: str
    emoji: str = "\U0001f4d2"


class UpdateNotebook(BaseModel):
    title: str | None = None
    emoji: str | None = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=12, ge=1, le=50)
    sort_by: str = "relevance"  # relevance | submittedDate


class DecideRequest(BaseModel):
    arxiv_ids: list[str]
    status: str = "confirmed"  # confirmed | rejected


class AskRequest(BaseModel):
    question: str
    mode: str = "dense"  # bm25 | dense | hybrid
    rerank: bool = False
    k: int = Field(default=5, ge=1, le=15)


# --- meta ----------------------------------------------------------------


@app.get("/api/health")
def health():
    """Tells the UI whether it is talking to real models or the offline
    fallbacks, because the fallbacks answer without ever saying so."""
    provider = get_provider()
    embedder = get_embedding_provider()
    reranker = get_reranker()
    live = type(provider).__name__ == "GeminiProvider"
    return {
        "ok": True,
        "live_models": live,
        "llm": {"provider": type(provider).__name__, "model": getattr(provider, "model", None)},
        "embeddings": {
            "provider": type(embedder).__name__,
            "model": getattr(embedder, "model", None),
        },
        "reranker": type(reranker).__name__,
        "notebooks": len(library.list()),
    }


# --- notebooks -----------------------------------------------------------


@app.get("/api/notebooks")
def list_notebooks():
    return [nb.summary() for nb in library.list()]


@app.post("/api/notebooks")
def create_notebook(body: CreateNotebook):
    return library.create(body.title, body.emoji).summary()


@app.get("/api/notebooks/{notebook_id}")
def get_notebook(notebook_id: str):
    nb = _nb(notebook_id)
    return {
        **nb.summary(),
        "sources": [s.model_dump() for s in nb.sources],
        "candidates": [c.model_dump() for c in nb.candidates],
        "chat": [t.model_dump() for t in nb.chat],
    }


@app.patch("/api/notebooks/{notebook_id}")
def update_notebook(notebook_id: str, body: UpdateNotebook):
    nb = _nb(notebook_id)
    if body.title is not None:
        nb.title = body.title
    if body.emoji is not None:
        nb.emoji = body.emoji
    return library.save(nb).summary()


@app.delete("/api/notebooks/{notebook_id}")
def delete_notebook(notebook_id: str):
    if not library.delete(notebook_id):
        raise HTTPException(404, f"no notebook {notebook_id}")
    invalidate(notebook_id)
    return {"deleted": notebook_id}


# --- sources -------------------------------------------------------------


@app.post("/api/notebooks/{notebook_id}/search")
def search(notebook_id: str, body: SearchRequest):
    """arXiv search, recorded as candidates so decisions persist."""
    nb = _nb(notebook_id)
    if not body.query.strip():
        raise HTTPException(400, "query is empty")
    try:
        papers = search_arxiv(body.query, max_results=body.max_results, sort_by=body.sort_by)
    except Exception as e:
        raise HTTPException(502, f"arXiv search failed: {e}") from e
    hits = library.add_candidates(nb, papers, body.query)
    return {"query": body.query, "results": [c.model_dump() for c in hits]}


@app.post("/api/notebooks/{notebook_id}/decide")
def decide(notebook_id: str, body: DecideRequest):
    if body.status not in ("confirmed", "rejected", "pending"):
        raise HTTPException(400, f"bad status {body.status}")
    nb = library.decide(_nb(notebook_id), body.arxiv_ids, body.status)
    invalidate(notebook_id)
    return {
        **nb.summary(),
        "sources": [s.model_dump() for s in nb.sources],
        "candidates": [c.model_dump() for c in nb.candidates],
    }


@app.delete("/api/notebooks/{notebook_id}/sources/{source_id}")
def remove_source(notebook_id: str, source_id: str):
    nb = library.remove_source(_nb(notebook_id), source_id)
    invalidate(notebook_id)
    return {
        **nb.summary(),
        "sources": [s.model_dump() for s in nb.sources],
        "candidates": [c.model_dump() for c in nb.candidates],
    }


# --- ask -----------------------------------------------------------------


@app.post("/api/notebooks/{notebook_id}/ask")
def ask(notebook_id: str, body: AskRequest):
    nb = _nb(notebook_id)
    if not nb.sources:
        raise HTTPException(400, "this notebook has no confirmed sources yet")
    if not body.question.strip():
        raise HTTPException(400, "question is empty")
    if body.mode not in ("bm25", "dense", "hybrid"):
        raise HTTPException(400, f"unknown retrieval mode {body.mode}")

    trace = Trace(name=f"notebook:{notebook_id}")
    started = time.time()
    turn = ChatTurn(
        turn_id=f"turn_{uuid.uuid4().hex[:8]}",
        question=body.question,
        mode=body.mode,
        rerank=body.rerank,
        trace_id=trace.trace_id,
    )
    try:
        retriever = get_retriever(nb, body.mode)
        claims = run_investigation(
            body.question,
            nb.sources,
            get_provider(),
            retriever,
            k=body.k,
            reranker=get_reranker() if body.rerank else None,
            trace=trace,
        )
        turn.claims = claims
    except Exception as e:
        # a provider 503 / quota wall should land in the transcript as a failed
        # turn, not vanish into a 500 the user has to guess at
        turn.error = f"{type(e).__name__}: {e}"
    turn.latency_ms = (time.time() - started) * 1000

    library.add_turn(nb, turn)
    if turn.error:
        raise HTTPException(502, turn.error)
    return turn.model_dump()


@app.get("/api/notebooks/{notebook_id}/sources/{source_id}/chunks")
def source_chunks(notebook_id: str, source_id: str):
    """Backs the citation drill-down: click an evidence chip, read the chunk in
    the context of its whole source."""
    nb = _nb(notebook_id)
    for s in nb.sources:
        if s.source_id == source_id:
            return {"source": s.model_dump()}
    raise HTTPException(404, f"no source {source_id}")
