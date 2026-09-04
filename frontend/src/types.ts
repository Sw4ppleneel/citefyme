export type CitationState = 'supported' | 'partial' | 'unsupported' | 'conflicting'
export type RetrievalMode = 'bm25' | 'dense' | 'hybrid'
export type CandidateStatus = 'pending' | 'confirmed' | 'rejected'

export interface Chunk {
  chunk_id: string
  source_id: string
  section: string
  text: string
}

export interface Source {
  source_id: string
  title: string
  authors: string[]
  year: number
  chunks: Chunk[]
}

export interface Evidence {
  evidence_id: string
  chunk_id: string
  source_id: string
  section: string
  text: string
}

export interface Claim {
  claim_id: string
  text: string
  evidence: Evidence[]
  citation_state: CitationState | null
  verification_note: string | null
}

export interface Candidate {
  arxiv_id: string
  title: string
  authors: string[]
  year: number
  abstract: string
  url: string
  query: string
  status: CandidateStatus
  source_id: string | null
  decided_at: number | null
}

export interface ChatTurn {
  turn_id: string
  question: string
  claims: Claim[]
  mode: RetrievalMode
  rerank: boolean
  latency_ms: number
  trace_id: string | null
  created_at: number
  error: string | null
}

export interface NotebookSummary {
  notebook_id: string
  title: string
  emoji: string
  created_at: number
  updated_at: number
  source_count: number
  chunk_count: number
  pending_count: number
  turn_count: number
}

export interface NotebookDetail extends NotebookSummary {
  sources: Source[]
  candidates: Candidate[]
  chat: ChatTurn[]
}

export interface Health {
  ok: boolean
  live_models: boolean
  llm: { provider: string; model: string | null }
  embeddings: { provider: string; model: string | null }
  reranker: string
  notebooks: number
}
