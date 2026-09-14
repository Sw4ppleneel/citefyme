"""Build a gold eval set for an arbitrary corpus.

Hand-labelling relevant chunks doesn't scale past a toy corpus, so questions
are generated *from* a known chunk: the chunk that seeded a question is its
gold-relevant chunk. Two things this buys, and one it doesn't:

  + labels are exact (we know which passage the question came from)
  + it works on any corpus, so evals aren't pinned to one hand-written set
  - questions are easier than real user questions, and lexically closer to the
    source passage than a real query would be, which flatters BM25.

The generated set is cached so numbers are reproducible across runs.
"""

import json
import os
from pathlib import Path

from citefyme.eval.dataset import EvalQuestion
from citefyme.llm import LLMProvider, _extract_json
from citefyme.models import Chunk, Source

GOLDSET_DIR = Path(os.environ.get("CITEFYME_HOME", str(Path.home() / ".citefyme"))) / "goldsets"

_QUESTION_PROMPT = """Passage from the paper "{title}" ({year}):
"{text}"

Write ONE specific research question that this passage answers.
Rules:
- It must be answerable from this passage alone, and it must be specific enough
  that a different paper's abstract would NOT answer it.
- Phrase it the way a researcher would ask it, and paraphrase: do NOT reuse the
  passage's distinctive multi-word phrases verbatim.
- Also give 2-4 short lowercase keywords (single words or word stems) that any
  correct answer must contain.

Respond ONLY with JSON, no markdown fences:
{{"question": "...", "keywords": ["...", "..."]}}"""

_UNANSWERABLE_PROMPT = """These are the titles of every paper in a corpus. The corpus
contains ONLY {content_desc} of these papers, nothing else:

{titles}

Write {n} questions that this corpus genuinely CANNOT answer, but that sound like
they should be answerable from it: ask for precise details this content never
contains (exact hyperparameters, GPU hours, per-benchmark numbers, ablation
details, dataset sizes) about papers that ARE in the list.

Respond ONLY with a JSON array of question strings, no markdown fences."""


def goldset_path(slug: str) -> Path:
    return GOLDSET_DIR / f"{slug}.json"


def _seed_chunks(sources: list[Source], n: int) -> list[Chunk]:
    """One chunk per source (the longest, i.e. most substantive), round-robin
    across sources so no single paper dominates the question set.

    Excludes the 'preamble' section (title/authors/affiliations/DOI
    boilerplate on full-text sources) from consideration: it's dense enough
    to often be the single longest chunk in a paper without containing any
    content a question should be seeded from. Falls back to preamble only if
    a source genuinely has nothing else."""
    per_source = []
    for s in sources:
        if not s.chunks:
            continue
        candidates = [c for c in s.chunks if c.section.strip().lower() != "preamble"]
        per_source.append(max(candidates or s.chunks, key=lambda c: len(c.text)))
    return per_source[:n]


def generate_goldset(
    sources: list[Source],
    provider: LLMProvider,
    n_answerable: int = 8,
    n_unanswerable: int = 3,
    content_desc: str = "the abstracts",
) -> tuple[list[EvalQuestion], list[EvalQuestion]]:
    by_id = {s.source_id: s for s in sources}

    answerable: list[EvalQuestion] = []
    for chunk in _seed_chunks(sources, n_answerable):
        src = by_id[chunk.source_id]
        raw = provider.complete(
            _QUESTION_PROMPT.format(title=src.title, year=src.year, text=chunk.text)
        )
        parsed = _extract_json(raw, "{", "}")
        if not parsed or not parsed.get("question"):
            continue
        answerable.append(
            EvalQuestion(
                question=parsed["question"].strip(),
                relevant_chunk_ids=[chunk.chunk_id],
                expected_keywords=[k.lower() for k in parsed.get("keywords", [])][:4],
            )
        )

    titles = "\n".join(f"- {s.title}" for s in sources)
    raw = provider.complete(
        _UNANSWERABLE_PROMPT.format(titles=titles, n=n_unanswerable, content_desc=content_desc)
    )
    questions = _extract_json(raw, "[", "]") or []
    unanswerable = [
        EvalQuestion(question=q.strip(), relevant_chunk_ids=[], answerable=False)
        for q in questions
        if isinstance(q, str)
    ][:n_unanswerable]

    if not answerable:
        raise RuntimeError(
            "gold-set generation produced no questions — needs a real LLM provider "
            f"(got {type(provider).__name__})"
        )

    return answerable, unanswerable


def get_goldset(
    slug: str,
    sources: list[Source],
    provider: LLMProvider,
    n_answerable: int = 8,
    n_unanswerable: int = 3,
    refresh: bool = False,
    content_desc: str = "the abstracts",
) -> tuple[list[EvalQuestion], list[EvalQuestion]]:
    path = goldset_path(slug)
    if path.exists() and not refresh:
        data = json.loads(path.read_text())
        return (
            [EvalQuestion(**q) for q in data["answerable"]],
            [EvalQuestion(**q) for q in data["unanswerable"]],
        )

    answerable, unanswerable = generate_goldset(
        sources, provider, n_answerable, n_unanswerable, content_desc
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "answerable": [q.model_dump() for q in answerable],
                "unanswerable": [q.model_dump() for q in unanswerable],
            },
            indent=2,
        )
    )
    return answerable, unanswerable
