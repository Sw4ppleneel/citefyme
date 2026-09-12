# CitefyMe

A local-first RAG system for asking questions over a corpus of papers where **every claim in the answer carries its supporting passage and a verification verdict** — not just a citation, but a labelled judgment of whether the source actually backs the claim.

Pipeline: ingest → chunk → retrieve (BM25 / dense / hybrid) → extract claims → verify each claim against its cited passage → rerank. A FastAPI backend and a small React/TypeScript UI sit on top of the same pipeline the eval harness uses, so the numbers below describe exactly what the UI does.

## Why

Most RAG demos stop at "here's an answer with some sources." CitefyMe treats citation as a first-class, checkable object: `Source → Chunk → Evidence → Claim`. Every claim the system makes is traceable to a specific chunk, and that chunk is run back through an entailment check (`supported` / `partial` / `unsupported`) instead of being trusted just because retrieval returned it.

Building it also meant measuring it — `results/` and `STACK.md` contain full eval output, including places the first design turned out to be wrong (see [Findings](#findings) below).

## Architecture

| Layer | Choice | Why |
|---|---|---|
| Data models | `pydantic` | `Source → Chunk → Evidence → Claim` — the core object graph |
| Chunking | hand-rolled (`citefyme/ingest.py`) | splits on markdown `## Section` headers, then windows long sections (400 chars, 60 overlap) so section context survives into every chunk |
| Sparse retrieval | `rank_bm25` | zero-dependency, no API key, exact-terminology matching |
| Dense retrieval | Gemini `gemini-embedding-001`, cosine similarity in `numpy` | falls back to a hashed bag-of-words vector with no key set, so the code path stays testable offline |
| Hybrid retrieval | Reciprocal Rank Fusion (RRF, k=60) over BM25 + dense | needs no score normalization between BM25 and cosine scores |
| Claim extraction + verification | Gemini (`GeminiProvider`), swappable via a `LLMProvider` interface (`AnthropicProvider`, deterministic `StubProvider`) | provider swaps don't touch pipeline code |
| Orchestration | plain Python (`citefyme/pipeline.py`), no agent framework | deterministic workflow, selective LLM calls — cheaper and easier to eval |
| Eval | custom (`citefyme/eval/`) | Recall@K / MRR for retrieval; citation precision, completeness, unsupported-rate for claims |
| API | FastAPI (`api/main.py`) | thin adapter — the eval harness and the UI both answer questions through one `run_investigation` |
| Notebooks | one JSON per notebook under `~/.citefyme/notebooks/` | library index is derived by reading the directory, so it can't drift out of sync |
| Retriever cache | `citefyme/index_cache.py`, keyed on `(notebook_id, version, mode)` | rebuilding a dense index per question would burn the embedding quota; `Notebook.version` invalidates the cache on any source change |
| Frontend | React 18 + TypeScript + Vite, plain CSS | two screens and one fetch client didn't justify a state library or router |
| Real sources | arXiv Atom API via stdlib `urllib` (`citefyme/arxiv.py`) | no key, no extra dependency; results are cached so eval numbers don't drift between runs |
| Eval sets | chunk-seeded synthetic generation (`citefyme/eval/synthetic.py`) | a question generated *from* a chunk has that chunk as its exact gold label — doesn't require hand-labelling |

## Setup

```bash
git clone https://github.com/Sw4ppleneel/citefyme.git
cd citefyme
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # paste in GEMINI_API_KEY (optional — see below)
```

### CLI / eval harness

```bash
.venv/bin/python demo.py hybrid                       # or: bm25 / dense
.venv/bin/python run_evals.py                         # 4 hand-written sample papers
.venv/bin/python run_topic_eval.py tiny-world-models  # 20 real arXiv papers
.venv/bin/python run_topic_eval.py rag-methods
```

### Notebook UI (two processes)

```bash
.venv/bin/uvicorn api.main:app --reload --port 8002   # API
cd frontend && npm install && npm run dev             # UI on http://localhost:5174
```

With no `GEMINI_API_KEY` set, everything still runs — `get_provider()` / `get_embedding_provider()` fall back to a deterministic stub provider and a hashed embedding provider, and the UI shows a banner saying so rather than silently answering as if it were the real model.

## Findings

Numbers below are from `run_topic_eval.py` against two real 20-paper arXiv corpora (full logs in `results/`), not the 4-paper toy set that shipped first:

| corpus | bm25 mrr | dense mrr | hybrid mrr | hybrid+rerank mrr |
|---|---|---|---|---|
| 4 sample papers | 0.92 | 0.94 | 1.00 | 1.00 |
| 20 arXiv, tiny world models | 0.75 | **0.94** | 0.79 | 0.94 |
| 20 arXiv, RAG methods | 0.88 | **1.00** | 0.88 | 0.94 |

- **Hybrid RRF underperforms dense alone.** Equal-weight RRF lets the weaker BM25 ranking drag good dense hits down the list on both real corpora; the LLM reranker's main measured effect is undoing that damage at the cost of an extra call per query. The UI defaults to dense retrieval with rerank off.
- **The 4-paper toy corpus hid this** — every mode scored 0.92–1.00 there, which can only tell you the code runs, not which retriever is better.
- **Retrieval makes the system more confidently wrong on unanswerable questions.** A no-retrieval baseline abstains far more often than the RAG path, which reliably finds *something* topically nearby and marks a claim `supported`. Citation coverage goes to 1.00, but "every claim has a citation" isn't "the system knows when to shut up" — abstention is a missing feature, not a tuning problem.
- **`unsupported_rate` is ~0.00 everywhere.** The verifier splits between `supported` and `partial` and has essentially never vetoed a claim outright — as a filter it currently downgrades, it doesn't reject.

Full methodology, a second benchmark run comparing against web-search agents, and the operational issues hit along the way (Gemini free-tier daily/per-minute caps, a `503` that escaped retry handling) are in [`STACK.md`](./STACK.md).

## What's real vs. a stand-in

**Real:** the full pipeline runs end-to-end against a live Gemini key (retrieval, claim extraction, entailment verification, reranking, tracing); both eval corpora are pulled live from the arXiv API, not hand-written; gold question sets are generated from the corpus rather than fixed.

**Still a stand-in:** corpora are abstracts only (no method/results sections to verify quantitative claims against); gold questions are chunk-seeded synthetic, which flatters retrieval relative to real user questions; the no-key fallback providers are sanity checks of the code, not quality numbers.

**Explicitly out of scope:** evidence graph / contradiction detection, a conflicting-citation state, a research planner / subquestion decomposition, a cross-encoder reranker, full-text (PDF/HTML) ingestion, non-arXiv sources, streaming responses, and abstention control.

## Project layout

```
citefyme/           core library — models, ingest, retrieval, hybrid, embeddings,
                     llm providers, pipeline, reranker, eval/, arxiv, library, cache
api/                 FastAPI app exposing notebooks/search/ask over the pipeline
frontend/            React + TypeScript notebook UI (Vite)
results/              raw output + logs from every eval/benchmark run
sample_data/          4 hand-written sample papers used by run_evals.py
STACK.md              full architecture notes, benchmark methodology, and findings
```
