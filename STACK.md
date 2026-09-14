# CitefyMe MVP — Stack & Architecture Notes

Scope of this build: chunking, retrieval (BM25 / dense / hybrid), claim
extraction + citation verification, and an eval harness. Everything below
is either implemented and tested, or explicitly flagged as a stand-in.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Data models | `pydantic` | `Source → Chunk → Evidence → Claim`, matches the product spec's core object graph |
| Chunking | hand-rolled (`citefyme/ingest.py`) | splits on markdown `## Section` headers, then windows long sections (400 chars, 60 overlap) so structure (section name) survives into every chunk |
| Sparse retrieval | `rank_bm25` (`BM25Okapi`) | zero-dependency, no API key, exact-terminology matching |
| Dense retrieval | Gemini `gemini-embedding-001` via `google-genai`, cosine similarity in `numpy` | free-tier embeddings; falls back to a hashed bag-of-words vector (`HashEmbeddingProvider`) when no key is set, so the dense code path is still exercisable offline |
| Hybrid retrieval | Reciprocal Rank Fusion (RRF, k=60) over BM25 + dense rankings | simplest fusion method that needs no score normalization between BM25 and cosine scores |
| Claim extraction + verification | Gemini `gemini-3.5-flash-lite` via `google-genai` (`GeminiProvider`), override with `GEMINI_MODEL` | the constraint for this build; also has an `AnthropicProvider` (unused by default) and a `StubProvider` (deterministic, lexical-overlap "verification") behind the same `LLMProvider` interface, so provider swaps don't touch pipeline code |
| Orchestration | plain Python functions (`citefyme/pipeline.py`), no agent framework | matches the "deterministic workflow, selective LLM calls" decision from the product spec — cheaper and easier to eval |
| Eval | custom (`citefyme/eval/`) | Recall@K / MRR for retrieval, citation precision / completeness / unsupported-rate for claims |
| API | `fastapi` + `uvicorn` (`api/main.py`) | thin adapter over the same pipeline — the eval harness and the UI answer questions through one `run_investigation` |
| Notebooks | one JSON per notebook under `~/.citefyme/notebooks/` (`citefyme/library.py`) | multi-topic version of `store.NotebookStore`; the library index is derived by reading the directory, so no index file can drift out of sync |
| Retriever cache | `citefyme/index_cache.py`, keyed on `(notebook_id, version, mode)` | a dense retriever embeds every chunk; rebuilding per question would burn the 1000 items/day embed quota. `Notebook.version` bumps on any source change, which invalidates the entry for free |
| Frontend | React 18 + TypeScript + Vite, plain CSS, no state library (`frontend/`) | two screens and one fetch client did not justify Zustand/router deps; hash routing keeps refresh-in-place |
| Real sources | arXiv Atom API via stdlib `urllib` (`citefyme/arxiv.py`) | no key, no extra dep; abstracts only, cached to `~/.citefyme/corpora/<topic>.json` so eval numbers don't drift between runs |
| Eval sets | chunk-seeded synthetic generation (`citefyme/eval/synthetic.py`) | hand-labelling relevant chunks doesn't scale past a toy corpus; a question generated *from* a chunk has that chunk as its exact gold label |
| Secrets | `.env` + `python-dotenv` | `GEMINI_API_KEY` (see `.env.example`), optional `GEMINI_MODEL` / `GEMINI_EMBEDDING_MODEL` |

## How to run

```bash
cd CitefyMe
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # then paste in GEMINI_API_KEY
.venv/bin/python demo.py hybrid        # or: bm25 / dense
.venv/bin/python run_evals.py                        # 4 hand-written sample papers
.venv/bin/python run_topic_eval.py tiny-world-models # 20 real arXiv papers
.venv/bin/python run_topic_eval.py rag-methods
```

To run the notebook UI (two processes):

```bash
cd CitefyMe
.venv/bin/uvicorn api.main:app --reload --port 8002   # API
cd frontend && npm install && npm run dev             # UI on http://localhost:5174
```

8002/5174 because 8000/8001/5173 are taken by the other projects in this
workspace; the API's CORS allowlist and Vite's `/api` proxy are both pinned to
those ports. With no `GEMINI_API_KEY` the UI still runs and says so in a
banner, because the stub providers otherwise answer without ever admitting
they are stubs.

`run_topic_eval.py` fetches a corpus from arXiv (topics defined in
`citefyme/topics.py`), generates a gold set, runs the whole eval battery, and
caches corpus + gold set under `~/.citefyme/` so re-runs measure the same
thing. `--sections retrieval,citation` runs a subset — useful because the
Gemini free tier caps at 500 generate requests/day *per model*, and a full
two-topic run lands right at that ceiling. When that cap is hit,
`call_with_backoff` now fails immediately with `DailyQuotaExhausted` instead
of retrying a quota that only resets tomorrow; switching `GEMINI_MODEL` gets a
fresh per-model allowance.

Without `GEMINI_API_KEY` set, everything still runs — `get_provider()` and
`get_embedding_provider()` silently fall back to `StubProvider` /
`HashEmbeddingProvider` (see `citefyme/llm.py`, `citefyme/embeddings.py`).
This was intentional so the harness is testable without a live key, but it
means retrieval and citation numbers from the fallback path are **sanity
checks of the code, not real quality numbers**.

## What's real vs. a stand-in (as of this build)

**Real, verified by running it against a live Gemini key:**
- The whole pipeline runs end-to-end on live `gemini-3.5-flash-lite` +
  `gemini-embedding-001`: retrieval, claim extraction, entailment
  verification, reranking, tracing. The earlier note that `GeminiProvider`'s
  request shapes were "verified by introspection, not a live call" is now
  obsolete — they work.
- Real corpora, not hand-written ones: `citefyme/arxiv.py` pulls papers from
  the arXiv API, and two 20-paper corpora (tiny/efficient world models, RAG
  methods + evaluation) are cached under `~/.citefyme/corpora/`.
- Gold sets are generated from the corpus rather than hand-written
  (`citefyme/eval/synthetic.py`), so the harness is no longer pinned to one
  toy question set.

**What the numbers actually said** (8 synthetic questions + 3 unanswerable
per corpus; full output in each run's log):

| corpus | bm25 mrr | dense mrr | hybrid mrr | hybrid+rerank mrr |
|---|---|---|---|---|
| 4 sample papers | 0.92 | 0.94 | 1.00 | 1.00 |
| 20 arXiv, tiny world models | 0.75 | **0.94** | 0.79 | 0.94 |
| 20 arXiv, RAG methods | 0.88 | **1.00** | 0.88 | 0.94 |

Four things fall out of this, and all four are arguments against choices
made in the first build:

1. **Hybrid RRF is not the best retriever — dense alone is.** On both real
   corpora, fusing BM25 into dense made retrieval *worse* than dense on its
   own (0.79 vs 0.94; 0.88 vs 1.00). RRF weights both rankings equally, and
   BM25 is the weaker signal on abstract-length chunks, so it drags good dense
   hits down the list. The LLM reranker's main measured effect is undoing that
   damage: hybrid+rerank lands at or below plain dense on both corpora, at the
   cost of an extra LLM call per query. Weighted RRF, or just defaulting to
   dense, is the cheap fix.
2. **The toy corpus was hiding this.** On the 4 sample papers every mode
   scored 0.92-1.00 and hybrid looked best. Retrieval eval on 8 questions over
   4 documents can only tell you the code runs.
3. **Retrieval makes the system *more* confidently wrong on questions the
   corpus can't answer.** Over the 3 unanswerable questions per corpus, the
   naive no-retrieval baseline abstains ("I don't have reliable information")
   far more than the RAG path, which finds *something* topically nearby and
   marks a claim `supported`:

   | corpus (model) | naive keyword_recall | RAG keyword_recall | naive halluc. | hybrid RAG halluc. |
   |---|---|---|---|---|
   | 4 sample papers (3.5-flash-lite) | 0.75 | 1.00 | 0.00 | 0.67 |
   | tiny world models (3.5-flash-lite) | 0.69 | 0.81 | 0.00 | 0.33 |
   | RAG methods (3.1-flash-lite) | 0.45 | 0.73 | 0.33 | 1.00 |

   Retrieval clearly wins on answer quality and gives 1.00 citation
   availability against 0.00 for the baseline — that part of the pitch holds.
   But "every claim has a citation" is not "the system knows when to shut
   up", and the citation-verification step does not currently supply the
   second one. Abstention is the missing feature, not a tuning problem.
   (Rows are internally consistent but **not comparable across rows**: the
   generator model differs, because the RAG-methods row had to be re-run on a
   different model after hitting the daily cap. n=3 is also tiny.)

4. **`unsupported_rate` is 0.00 everywhere.** Across every mode and every
   corpus, the verifier never returned `unsupported` — it splits between
   `supported` (~0.73-0.87 citation precision) and `partial`, and `partial` is
   where all the disagreement goes. As a filter it currently rejects nothing;
   its real output is a downgrade, not a veto.

**Still a stand-in / still not trustworthy:**
- Corpora are **abstracts only**. Real claim verification needs the method and
  results sections; an abstract can support "X was proposed" but rarely "X
  achieved Y".
- Gold questions are **chunk-seeded and synthetic**: generated from the chunk
  they are labelled against. That makes labels exact but questions easier and
  more lexically similar to the source than real queries, which flatters
  retrieval across the board. Real user questions will score lower.
- 3 unanswerable questions per corpus is too small to read a hallucination
  rate off — the direction changed between corpora and between models.
- `StubProvider` / `HashEmbeddingProvider` remain no-key fallbacks and their
  numbers are still only code sanity checks.

**Operational things the runs exposed:**
- The Gemini free tier caps generate at 500 requests/day *per model*, embed at
  100 items/min and 1000 items/day *per embedding model*; a full two-topic
  eval sits right at both day caps.
  `--sections` exists so a section lost to the cap can be re-run alone, and
  switching `GEMINI_MODEL` buys a fresh daily allowance.
- Batched `embed_content` bills **per item**, not per HTTP call, so the
  embed limiter consumes `len(texts)` slots, and `build_retriever` now hands
  the dense index to the hybrid retriever instead of embedding the corpus
  twice.

Raw output from every run above is checked in under `results/`.

## Benchmark run, 2026-09-04: three arms on the `rag-methods` corpus

Same fixed gold set for every arm (8 chunk-labelled answerable + 3
deliberately unanswerable questions, 20 arXiv papers, 84 chunks), and the same
scoring functions — the web-search arm is graded by
`score_search_agents.py`, which imports `keyword_recall` /
`expresses_uncertainty` from `citefyme/eval/baseline_compare.py` rather than
reimplementing them. Logs: `results/2026-09-04_*`.

**Arm 3 — retrieval technique (no LLM in the metric except rerank):**

| mode | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| bm25 | 0.88 | 0.88 | 0.88 | 0.88 |
| **dense** | **1.00** | **1.00** | **1.00** | **1.00** |
| hybrid (RRF) | 0.88 | 0.88 | 0.88 | 0.88 |
| hybrid+rerank | 1.00 | 1.00 | 1.00 | 1.00 |

This sharpens finding 1 above rather than repeating it: hybrid RRF now scores
*identically to bm25 alone*, i.e. fusing the weaker sparse signal in erased
dense's entire advantage, and the LLM reranker's only measured job was buying
that loss back at one extra call per query. Dense alone gets there for free.
The UI therefore defaults to dense with rerank off.

**Arms 1 and 2 — direct vs RAG** (both on `gemini-3.1-flash-lite`, so these
rows are comparable to each other):

| condition | keyword_recall | citation_availability | hallucination_rate |
|---|---|---|---|
| naive LLM, no retrieval | 0.62 | 0.00 | 0.33 |
| hybrid RAG | 0.77 | 1.00 | **1.00** |
| hybrid RAG + rerank | 0.80 | 1.00 | **1.00** |
| web search agents | 1.00 | 1.00 | **0.00** |

Claim-level citation quality by mode: bm25 1.00 precision, hybrid 0.94,
hybrid+rerank 0.90, all at 1.00 completeness. `unsupported_rate` is still
~0.00 (0.05 for hybrid+rerank) — the verifier continues to downgrade to
`partial` rather than veto, so as a filter it still rejects almost nothing.

Two things to be careful about before quoting the table:

1. **The abstention gap is the headline, and it got worse, not better.** RAG
   answers *better* on answerable questions (0.62 -> 0.77/0.80) and gives
   citations where the baseline gives none, but it asserted a `supported`
   claim on **all three** unanswerable questions where the no-retrieval
   baseline abstained on two. Retrieval always finds something topically
   nearby, and nothing in the pipeline is allowed to say "not in this corpus".
   This is the same conclusion as before at a worse number (0.33 -> 1.00),
   which is an argument for abstention being a missing *feature*, not drift.
2. **The web-search arm's sweep is not a like-for-like win.** It is a
   different model (Claude subagents) *and* a different corpus (the live web,
   including the full papers rather than abstracts), so it confounds arm with
   model. What it does isolate is the abstention behaviour: those agents
   scored 0.00 hallucination because they could open each paper and verify the
   asked-for number was absent — one of the three questions turns out to have
   a false premise (the DataMorgana paper reports no F1 scores at all). No
   abstract-only corpus can support that judgement, which is a concrete
   argument for full-text ingestion.
3. **keyword_recall saturates.** Every answerable question scored exactly 1.00
   in the search arm because the metric is a lexical substring check and an
   agent that finds the right paper trivially echoes its terminology. Treat
   1.00 as "found the source", not "answered well" — the metric has no
   headroom left and should be replaced before it is used to compare anything
   again.

**Operational fix this run forced:** `call_with_backoff` only caught
`ClientError`, so a Gemini `503 UNAVAILABLE` (a `ServerError`) escaped it
entirely and killed an hour-long run part-way through a section — twice. It
now retries 5xx with the same backoff. Sustained overload on one model still
needs `GEMINI_MODEL` switched, which is how the citation/baseline sections
above were finally completed.

## Full-text ingestion, 2026-09-14: `electrogels-prosthetics`

First run against whole papers instead of abstracts (`citefyme/fulltext.py`,
`Topic.fulltext=True`). arXiv's HTML5 rendering (LaTeXML, at
`arxiv.org/html/<id>`) is parsed for 3 of the 4 papers; the fourth has no
HTML rendering and falls back to `pypdf` text extraction from the PDF.
Corpus: 4 papers, 561 chunks (vs. ~20-84 chunks for 20 *abstracts* in the
other topics — one full paper chunks to roughly the size of a whole abstract
corpus). Log: `results/2026-09-14_electrogels-prosthetics_bm25.log`.

**Topic note:** arXiv has essentially no materials-chemistry papers literally
branded "electrogel" — that literature (ionically/PEDOT/MXene-conductive
hydrogels) lives in journals like *Advanced Materials* and RSC, not arXiv.
The corpus is the closest real cluster arXiv has: conductive-hydrogel-based
e-skin/tactile-sensing papers (the prosthetics-relevant application of the
material) plus a general robotic-prosthetics survey for domain context. Same
situation as `tiny-world-models`'s search-term note — arXiv coverage, not the
pipeline, is the constraint.

**Two real bugs this run found, both in code that had never seen a
full-text-sized corpus before:**

1. **`BatchEmbedContentsRequest` hard-caps at 100 items/call**, and
   `GeminiEmbeddingProvider.embed` was sending the whole corpus (561 texts)
   in one call. Abstract corpora topped out at 84 chunks and never hit it.
   Fixed by batching into groups of 90 (`citefyme/embeddings.py`).
2. **Gold-set seeding picked front matter, not content.** `_seed_chunks`
   chose "the longest chunk per source" as the most-substantive passage to
   seed a question from — true for a single-chunk abstract, false for a
   561-chunk paper. The HTML extractor was also treating the paper's `<h1>`
   title as a section heading, so every author/affiliation/DOI block got
   bucketed into a fake section named after the paper's own title; being
   dense unbroken text, it reliably won "longest chunk." First run's "gold"
   questions were things like *"At which academic institution... are Haofeng
   Chen, Bedrich Himmel, and Matej Hoffmann employed?"* — a real, exactly-
   labelled question, just not one that tests whether retrieval finds
   *hydrogel content*. Fixed two ways: `<h1>` is no longer treated as a
   heading boundary (front matter now buckets into `preamble`, matching the
   PDF-extraction path), and `_seed_chunks` excludes `preamble` from
   candidates. Re-running after the fix produced genuinely content-seeded
   questions (*"What technological limitations currently hinder... flexible
   physiological tracking?"*).

**BM25 results** (dense/hybrid blocked — see below):

| mode | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| bm25 | 0.50 | 0.50 | 0.50 | 0.50 |

| condition | keyword_recall | citation_availability | hallucination_rate |
|---|---|---|---|
| naive LLM, no retrieval | 0.69 | 0.00 | 0.00 |
| bm25 RAG | 0.69 | 1.00 | 0.67 |

Citation precision 0.71, completeness 1.00, unsupported_rate 0.00 — same
downgrade-not-veto pattern as the abstract-only corpora. Recall@1 of 0.50 on
4 answerable questions (n=4, so read this as a data point, not a stable
number) is the weakest BM25 score of any corpus run so far, consistent with
finding 1 from the earlier abstracts-only run: BM25 is the retriever most
exposed by paraphrased, non-lexically-matching questions, and full papers
give the gold-set generator far more paraphrase room than a two-sentence
abstract does.

**What's missing and why:** dense and hybrid modes are blocked by the Gemini
free tier's `EmbedContentRequestsPerDayPerProjectPerModel` cap (1000
items/day) — embedding 561 chunks twice (once before the seeding fix, once
after `--refresh-corpus`) burned the day's allowance before a third attempt
could complete. Unlike the per-minute cap, this doesn't clear by waiting a
few minutes, and this account has no working alternate embedding model to
switch to (`text-embedding-004` and `embedding-001` both 404 on this API
version — the `GEMINI_MODEL`-switch workaround documented above is for the
*generation* quota, which is a separate, model-swappable metric; the free
tier effectively offers one embedding model). `run_topic_eval.py` now takes
`--modes` (mirrors `--sections`) so a mode list can be run standalone once
the daily quota resets:
`run_topic_eval.py electrogels-prosthetics --modes dense,hybrid --sections retrieval,citation,baseline`.

## Explicitly out of scope for this build

Evidence graph / contradiction detection, citation-state ⚫ (conflicting),
research planner / subquestion decomposition, cross-encoder reranker model,
non-arXiv sources (Semantic Scholar/web), and the "research gaps" feature
from the full CitefyMe spec. The architecture
(provider seams, `Source → Chunk → Evidence → Claim` model, deterministic
pipeline) was built so all of those bolt on without rewrites — but none of
them exist yet.

The UI is no longer out of scope: `api/` + `frontend/` are a notebook-per-topic
front end over exactly this pipeline (arXiv search -> confirm/reject review
queue -> chat whose every claim carries its passage and verification state).
What it deliberately does *not* add is anything the pipeline cannot back:
there is no abstention control, so the hallucination_rate above is the UI's
behaviour too, and no streaming, so a chat turn blocks for the full
retrieve/extract/verify round trip (~15-60s on a warm index, longer while the
free tier is rate-limiting or 503-ing).
