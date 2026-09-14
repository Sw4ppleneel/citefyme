"""Full-text ingestion: fetch whole-paper text (not just the abstract) and
turn it into the same Source/Chunk shape the rest of the pipeline expects.

Two paths, tried in order:
1. arXiv's auto-generated HTML5 rendering (via LaTeXML), at
   https://arxiv.org/html/<id> — available for most papers submitted since
   Dec 2023. We strip that down to '## Heading' + plain-text paragraphs so
   citefyme/ingest.py's existing section-aware chunker works unchanged.
2. The PDF, via `pypdf`, for papers with no HTML rendering (older
   submissions, or ones that opted out). Section headings are recovered with
   a keyword heuristic (short line, matches a known academic section name)
   since a PDF carries no structural markup; text is truncated at the first
   references/supplementary-material heading so bibliographies don't get
   chunked as if they were prose.

Only falls back to the abstract-only Source (citefyme/arxiv.py) if both fail.
"""

import io
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser

from pypdf import PdfReader

from citefyme.arxiv import _slug, paper_to_source, search_arxiv
from citefyme.ingest import build_source
from citefyme.models import Source

HTML_URL = "https://arxiv.org/html/{arxiv_id}"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

# Tags whose entire subtree contributes no usable prose (nav chrome, markup
# tables, raw MathML/SVG source that would otherwise leak into chunk text).
_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "table", "math", "svg"}
# h1 is the paper's own title (LaTeXML: <h1 class="ltx_title_document">), not a
# section boundary — treating it as one bucketed the whole front matter
# (authors, affiliations, DOI/conference boilerplate) into a fake "section"
# named after the title, which is dense enough to look like the paper's most
# substantive chunk to anything picking "the longest chunk" as a seed.
_HEADING_TAGS = {"h2", "h3", "h4"}
# LaTeXML marks the bibliography with these classes; citations there are a
# wall of author/year strings, not prose worth chunking or citing.
_SKIP_CLASSES = ("ltx_bibliography",)


class _SectionTextExtractor(HTMLParser):
    """Walks LaTeXML's HTML, emitting a '## Heading' line before each
    section heading and plain text everywhere else."""

    def __init__(self):
        super().__init__()
        self._stack: list[tuple[str, bool]] = []
        self._in_heading = False
        self._heading_buf: list[str] = []
        self.parts: list[tuple[str, str]] = []  # ("text" | "heading", value)

    def _should_skip(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in _SKIP_TAGS:
            return True
        attr_dict = dict(attrs)
        if "hidden" in attr_dict:
            return True
        classes = attr_dict.get("class") or ""
        return any(c in classes for c in _SKIP_CLASSES)

    def _currently_skipping(self) -> bool:
        return any(skip for _, skip in self._stack)

    def handle_starttag(self, tag, attrs):
        skip = self._currently_skipping() or self._should_skip(tag, attrs)
        self._stack.append((tag, skip))
        if not skip and tag in _HEADING_TAGS:
            self._in_heading = True
            self._heading_buf = []

    def handle_endtag(self, tag):
        was_skipping = self._currently_skipping()
        if self._stack:
            self._stack.pop()
        if not was_skipping and tag in _HEADING_TAGS and self._in_heading:
            title = " ".join("".join(self._heading_buf).split())
            title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", title)  # strip "3.2 " numbering
            if title:
                self.parts.append(("heading", title))
            self._in_heading = False

    def handle_data(self, data):
        if self._currently_skipping():
            return
        if self._in_heading:
            self._heading_buf.append(data)
        else:
            self.parts.append(("text", data))

    def markdown(self) -> str:
        out = []
        for kind, val in self.parts:
            out.append(f"\n## {val}\n" if kind == "heading" else val)
        text = "".join(out)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def fetch_fulltext_html(arxiv_id: str, timeout: int = 60) -> str | None:
    """Raw arXiv HTML5 rendering for a paper, or None if arXiv hasn't
    generated one (404 — older paper, or opted out of HTML)."""
    base_id = arxiv_id.split("v")[0]
    req = urllib.request.Request(
        HTML_URL.format(arxiv_id=base_id),
        headers={"User-Agent": "citefyme/0.1 (research prototype)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def html_to_sections(html: str) -> str:
    """Parse only the <article class="ltx_document">...</article> body —
    everything outside it is arXiv page chrome (nav, theme toggle, a
    'report an issue' modal, related-paper recommendations) that has nothing
    to do with the paper itself."""
    match = re.search(r"<article\b[^>]*>.*</article>", html, re.DOTALL)
    fragment = match.group(0) if match else html
    parser = _SectionTextExtractor()
    parser.feed(fragment)
    return parser.markdown()


# Recognized academic section names, matched against short standalone lines
# in PDF-extracted text (which carries no tag structure to key off).
_PDF_SECTION_KEYWORDS = (
    "abstract", "introduction", "background", "related work",
    "materials and methods", "materials & methods", "methods", "methodology",
    "experimental section", "experimental", "results", "results and discussion",
    "discussion", "conclusion", "conclusions", "acknowledgements", "acknowledgments",
)
# Headings that mark the end of citable prose — everything from here on
# (bibliography, SI) is truncated rather than chunked.
_PDF_STOP_KEYWORDS = ("references", "supplementary", "supporting information")


def _pdf_heading(line: str) -> str | None:
    if not line or len(line) > 60:
        return None
    stripped = re.sub(r"^[\dIVXivx]+[.\)]?\s*", "", line).strip().rstrip(":")
    low = stripped.lower()
    if low in _PDF_SECTION_KEYWORDS:
        return stripped
    return None


def fetch_fulltext_pdf(arxiv_id: str, timeout: int = 60) -> bytes | None:
    """Raw PDF bytes for a paper, trying the given id and, if that 404s, the
    'v1' revision (arXiv's /pdf/ endpoint wants a version suffix)."""
    for candidate in (arxiv_id, arxiv_id if "v" in arxiv_id else f"{arxiv_id}v1"):
        req = urllib.request.Request(
            PDF_URL.format(arxiv_id=candidate),
            headers={"User-Agent": "citefyme/0.1 (research prototype)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
    return None


def pdf_to_sections(pdf_bytes: bytes, max_chars: int = 45_000) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)

    out_lines: list[str] = []
    total = 0
    for line in raw.split("\n"):
        stripped = line.strip()
        low = stripped.lower().rstrip(":")
        if total > 500 and any(low.startswith(kw) for kw in _PDF_STOP_KEYWORDS):
            break
        heading = _pdf_heading(stripped)
        out_lines.append(f"## {heading}" if heading else stripped)
        total += len(stripped)
        if total >= max_chars:
            break
    return "\n".join(out_lines).strip()


def paper_to_fulltext_source(paper: dict, full_text: str, source_id: str | None = None) -> Source:
    source_id = source_id or f"arxiv_full_{_slug(paper['title'], 32)}"
    return build_source(
        source_id=source_id,
        title=paper["title"],
        authors=paper["authors"],
        year=paper["year"],
        raw_text=full_text,
    )


def fetch_fulltext_corpus(papers: list[str | dict], pause_s: float = 3.0) -> list[Source]:
    """One Source per paper, built from the full paper text: arXiv's HTML
    rendering if it exists, else the PDF via pypdf, else the abstract.

    Each entry is either a bare arXiv id (title/authors/year looked up via
    the arXiv search API) or a dict with those fields pre-supplied
    ({"arxiv_id", "title", "authors", "year"}) — useful when the id is
    already known and there's no reason to spend a search-API call on it.
    """
    sources = []
    for i, entry in enumerate(papers):
        if i:
            time.sleep(pause_s)
        if isinstance(entry, dict):
            paper = entry
            arxiv_id = entry["arxiv_id"]
        else:
            arxiv_id = entry
            matches = search_arxiv(f"id:{arxiv_id}", max_results=1)
            if not matches:
                raise ValueError(f"arXiv id not found: {arxiv_id}")
            paper = matches[0]
        sid = f"arxiv_full_{_slug(paper['title'], 32)}"

        html = fetch_fulltext_html(arxiv_id)
        if html is not None:
            full_text = html_to_sections(html)
            print(f"  [{arxiv_id}] full text from HTML rendering")
            sources.append(paper_to_fulltext_source(paper, full_text, sid))
            continue

        pdf_bytes = fetch_fulltext_pdf(arxiv_id)
        if pdf_bytes is not None:
            full_text = pdf_to_sections(pdf_bytes)
            print(f"  [{arxiv_id}] full text from PDF ({len(full_text)} chars)")
            sources.append(paper_to_fulltext_source(paper, full_text, sid))
            continue

        print(f"  [{arxiv_id}] no HTML or PDF full text found — using abstract only")
        sources.append(paper_to_source(paper))
    return sources
