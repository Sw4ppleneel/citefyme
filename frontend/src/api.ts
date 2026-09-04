import type {
  Candidate,
  CandidateStatus,
  ChatTurn,
  Health,
  NotebookDetail,
  NotebookSummary,
  RetrievalMode,
  Source,
} from './types'

/** Vite proxies /api to the FastAPI server on 8002 (see vite.config.ts). */
async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'content-type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    // FastAPI puts the message in `detail`; surface it rather than a bare status
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => call<Health>('/health'),

  listNotebooks: () => call<NotebookSummary[]>('/notebooks'),

  createNotebook: (title: string, emoji: string) =>
    call<NotebookSummary>('/notebooks', {
      method: 'POST',
      body: JSON.stringify({ title, emoji }),
    }),

  getNotebook: (id: string) => call<NotebookDetail>(`/notebooks/${id}`),

  updateNotebook: (id: string, patch: { title?: string; emoji?: string }) =>
    call<NotebookSummary>(`/notebooks/${id}`, { method: 'PATCH', body: JSON.stringify(patch) }),

  deleteNotebook: (id: string) =>
    call<{ deleted: string }>(`/notebooks/${id}`, { method: 'DELETE' }),

  search: (id: string, query: string, maxResults = 12, sortBy = 'relevance') =>
    call<{ query: string; results: Candidate[] }>(`/notebooks/${id}/search`, {
      method: 'POST',
      body: JSON.stringify({ query, max_results: maxResults, sort_by: sortBy }),
    }),

  decide: (id: string, arxivIds: string[], status: CandidateStatus) =>
    call<NotebookDetail>(`/notebooks/${id}/decide`, {
      method: 'POST',
      body: JSON.stringify({ arxiv_ids: arxivIds, status }),
    }),

  removeSource: (id: string, sourceId: string) =>
    call<NotebookDetail>(`/notebooks/${id}/sources/${sourceId}`, { method: 'DELETE' }),

  ask: (id: string, question: string, mode: RetrievalMode, rerank: boolean, k = 5) =>
    call<ChatTurn>(`/notebooks/${id}/ask`, {
      method: 'POST',
      body: JSON.stringify({ question, mode, rerank, k }),
    }),

  sourceChunks: (id: string, sourceId: string) =>
    call<{ source: Source }>(`/notebooks/${id}/sources/${sourceId}/chunks`),
}
