from enum import Enum
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    section: str
    text: str


class Source(BaseModel):
    source_id: str
    title: str
    authors: list[str]
    year: int
    chunks: list[Chunk]


class CitationState(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"


class Evidence(BaseModel):
    evidence_id: str
    chunk_id: str
    source_id: str
    section: str
    text: str


class Claim(BaseModel):
    claim_id: str
    text: str
    evidence: list[Evidence]
    citation_state: CitationState | None = None
    verification_note: str | None = None
