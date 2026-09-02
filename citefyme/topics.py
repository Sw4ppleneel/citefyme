"""Topic definitions: a named research topic -> the arXiv queries that build
its corpus, plus the headline question to investigate."""

from dataclasses import dataclass, field


@dataclass
class Topic:
    slug: str
    label: str
    question: str
    queries: list[str]
    exclude_terms: tuple[str, ...] = ()
    per_query: int = 10
    limit: int = 20
    notes: str = ""


TOPICS: dict[str, Topic] = {
    "tiny-world-models": Topic(
        slug="tiny-world-models",
        label="Tiny / efficient world models",
        question=(
            "How do small, efficient world models stay accurate, and what do they "
            "give up compared to large ones?"
        ),
        queries=[
            'ti:"world model" AND (ti:tiny OR ti:small OR ti:compact OR ti:lightweight OR ti:efficient)',
            'abs:"world model" AND (abs:"sample-efficient" OR abs:"parameter-efficient" OR abs:distill)',
            'ti:"world models" AND (abs:tiny OR abs:small OR abs:compact)',
        ],
        # 'small world' also names a graph-theory topic with nothing to do with
        # learned world models; drop those rather than let them pollute recall.
        exclude_terms=(
            "small-world network", "small world network", "kleinberg",
            "contagion", "homeomorphic",
        ),
        notes="arXiv has no paper literally titled 'Tiny World Model'; this is the "
              "tiny/compact/efficient learned-world-model cluster.",
    ),
    "rag-methods": Topic(
        slug="rag-methods",
        label="RAG methods and evaluation",
        question=(
            "What retrieval-augmented generation methods reduce hallucination, and "
            "how is RAG quality actually evaluated?"
        ),
        queries=[
            'ti:"retrieval-augmented generation" AND (ti:survey OR abs:survey)',
            'ti:"retrieval-augmented generation" AND (ti:graph OR ti:agentic OR ti:adaptive OR ti:self)',
            'abs:"retrieval-augmented generation" AND (ti:reranking OR ti:chunking OR ti:hybrid OR ti:"query rewriting")',
            'ti:"RAG" AND (ti:hallucination OR ti:evaluation OR ti:benchmark)',
        ],
        per_query=8,
    ),
}
