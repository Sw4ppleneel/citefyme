"""Real source ingestion from the arXiv API.

Fetches paper metadata + abstracts for a search query and turns them into
`Source` objects with the same `Source -> Chunk` shape as the hand-written
sample corpus, so every retriever / eval path works unchanged.

Abstracts only, not full PDF text: an abstract is real, quotable, provenance-
carrying prose, and it keeps the corpus small enough to embed on a free-tier
key. Full-text (PDF/HTML) ingestion is the obvious next step.
"""

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from citefyme.ingest import build_source
from citefyme.models import Source

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n] or "paper"


def search_arxiv(query: str, max_results: int = 15, sort_by: str = "relevance") -> list[dict]:
    """Return raw paper dicts (arxiv_id, title, authors, year, abstract, url)."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
    )
    req = urllib.request.Request(
        f"{ARXIV_API}?{params}", headers={"User-Agent": "citefyme/0.1 (research prototype)"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        xml = resp.read()

    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("a:entry", NS):
        arxiv_url = entry.findtext("a:id", default="", namespaces=NS)
        arxiv_id = arxiv_url.rsplit("/", 1)[-1]
        title = " ".join(entry.findtext("a:title", default="", namespaces=NS).split())
        abstract = " ".join(entry.findtext("a:summary", default="", namespaces=NS).split())
        published = entry.findtext("a:published", default="", namespaces=NS)
        authors = [
            a.findtext("a:name", default="", namespaces=NS)
            for a in entry.findall("a:author", NS)
        ]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "year": int(published[:4]) if published[:4].isdigit() else 0,
                "abstract": abstract,
                "url": arxiv_url,
            }
        )
    return papers


def paper_to_source(paper: dict, source_id: str | None = None) -> Source:
    source_id = source_id or f"arxiv_{_slug(paper['title'], 32)}"
    raw = f"## Abstract\n{paper['abstract']}"
    return build_source(
        source_id=source_id,
        title=paper["title"],
        authors=paper["authors"],
        year=paper["year"],
        raw_text=raw,
    )


def fetch_corpus(query: str, max_results: int = 15, pause_s: float = 3.0) -> list[Source]:
    """arXiv asks for ~3s between API calls; one call per corpus here."""
    papers = search_arxiv(query, max_results=max_results)
    time.sleep(pause_s)
    seen: set[str] = set()
    sources = []
    for p in papers:
        sid = f"arxiv_{_slug(p['title'], 32)}"
        if sid in seen:
            sid = f"{sid}_{p['arxiv_id'].replace('.', '_')}"
        seen.add(sid)
        sources.append(paper_to_source(p, sid))
    return sources


def fetch_corpus_multi(
    queries: list[str],
    per_query: int = 10,
    exclude_terms: tuple[str, ...] = (),
    limit: int | None = None,
    pause_s: float = 3.0,
) -> list[Source]:
    """Union several arXiv queries into one deduped corpus.

    One query is rarely enough for a topic ('world model' alone drags in
    small-world *network* papers), so we OR a few phrasings together and drop
    anything whose title/abstract hits an exclude term.
    """
    papers: dict[str, dict] = {}
    for i, q in enumerate(queries):
        if i:
            time.sleep(pause_s)
        for p in search_arxiv(q, max_results=per_query):
            base_id = p["arxiv_id"].split("v")[0]
            if base_id in papers:
                continue
            blob = f"{p['title']} {p['abstract']}".lower()
            if any(t.lower() in blob for t in exclude_terms):
                continue
            papers[base_id] = p

    ordered = sorted(papers.values(), key=lambda p: p["year"], reverse=True)
    if limit:
        ordered = ordered[:limit]

    seen: set[str] = set()
    sources = []
    for p in ordered:
        sid = f"arxiv_{_slug(p['title'], 32)}"
        if sid in seen:
            sid = f"{sid}_{p['arxiv_id'].replace('.', '_')}"
        seen.add(sid)
        sources.append(paper_to_source(p, sid))
    return sources
