"""Topic definitions: a named research topic -> the arXiv queries that build
its corpus, plus the headline question to investigate."""

from dataclasses import dataclass, field


@dataclass
class Topic:
    slug: str
    label: str
    question: str
    queries: list[str] = field(default_factory=list)
    exclude_terms: tuple[str, ...] = ()
    per_query: int = 10
    limit: int = 20
    notes: str = ""
    # Full-text topics skip the query-based abstract search entirely and
    # ingest whole papers (see citefyme/fulltext.py) for a curated id list.
    fulltext: bool = False
    arxiv_ids: list[str | dict] = field(default_factory=list)


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
    "electrogels-prosthetics": Topic(
        slug="electrogels-prosthetics",
        label="Electrogels (conductive hydrogels) in prosthetics",
        question=(
            "How are conductive hydrogels ('electrogels') used in prosthetic and "
            "robotic tactile sensing, and what do they trade off against other "
            "e-skin materials?"
        ),
        fulltext=True,
        notes="arXiv has essentially no materials-chemistry papers literally branded "
              "'electrogel' — that literature lives in journals like Advanced "
              "Materials / RSC, not arXiv. This is the closest real cluster arXiv "
              "has: conductive-hydrogel-based e-skin/tactile sensing (the "
              "prosthetics-relevant application of the material) plus a general "
              "robotic-prosthetics survey for domain context.",
        arxiv_ids=[
            {
                "arxiv_id": "2408.01729",
                "title": "A Survey on Robotic Prosthetics: Neuroprosthetics, Soft "
                          "Actuators, and Control Strategies",
                "authors": ["Kumar J. Jyothish", "Subhankar Mishra"],
                "year": 2024,
            },
            {
                "arxiv_id": "2503.13048",
                "title": "Multi-Touch and Bending Sensing Using Electrical Impedance "
                          "Tomography for Robotics",
                "authors": ["Haofeng Chen", "Bedrich Himmel", "Bin Li", "Xiaojie Wang",
                            "Matej Hoffmann"],
                "year": 2025,
            },
            {
                "arxiv_id": "2504.05987",
                "title": "Learning-enhanced electronic skin for tactile sensing on "
                          "deformable surface based on electrical impedance tomography",
                "authors": ["Huazhi Dong", "Xiaopeng Wu", "Delin Hu", "Zhe Liu",
                            "Francesco Giorgio-Serchi", "Yunjie Yang"],
                "year": 2025,
            },
            {
                "arxiv_id": "2502.17208",
                "title": "A highly sensitive, self-adhesive, biocompatible DLP 3D "
                          "printed organohydrogel for flexible sensors and wearable "
                          "devices",
                "authors": ["Ze Zhang", "Kewei Song", "Kayo Hirose", "Jianxian He",
                            "Qianhao Li", "Yannan Li", "Yifan Pan", "Mohamed Adel",
                            "Rongyi Zhuang", "Shogo Iwai", "Ahmed M. R. Fath El-Bab",
                            "Hui Fang", "Zhouyuan Yang", "Shinjiro Umezu"],
                "year": 2025,
            },
        ],
    ),
}
