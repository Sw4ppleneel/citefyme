import re

from citefyme.models import Chunk, Source


def chunk_document(
    raw_text: str,
    source_id: str,
    max_chunk_chars: int = 400,
    overlap_chars: int = 60,
) -> list[Chunk]:
    """Split a markdown-style document ('## Section' headers) into chunks,
    preserving which section each chunk came from. Splits long sections
    into overlapping windows so no chunk exceeds max_chunk_chars."""
    sections = _split_by_headers(raw_text)

    chunks: list[Chunk] = []
    idx = 0
    for section_title, section_text in sections:
        for window in _windows(section_text, max_chunk_chars, overlap_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{source_id}_c{idx}",
                    source_id=source_id,
                    section=section_title,
                    text=window,
                )
            )
            idx += 1
    return chunks


def _split_by_headers(raw_text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^##\s+(.+)$", raw_text.strip())
    if len(parts) == 1:
        return [("body", raw_text.strip())]

    sections = []
    preamble = parts[0].strip()
    if preamble:
        sections.append(("preamble", preamble))
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sections.append((title, body))
    return sections


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    windows = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        windows.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return windows


def build_source(
    source_id: str, title: str, authors: list[str], year: int, raw_text: str
) -> Source:
    return Source(
        source_id=source_id,
        title=title,
        authors=authors,
        year=year,
        chunks=chunk_document(raw_text, source_id),
    )
