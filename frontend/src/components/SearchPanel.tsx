import { useMemo, useState } from 'react'
import type { Candidate, CandidateStatus } from '../types'

type Filter = 'pending' | 'confirmed' | 'rejected'

/** arXiv search + the confirm/reject review queue.
 *  A decision is remembered, so a rejected paper does not come back on the
 *  next search for the same topic. */
export default function SearchPanel({
  candidates,
  busy,
  onSearch,
  onDecide,
}: {
  candidates: Candidate[]
  busy: boolean
  onSearch: (query: string, sortBy: string) => Promise<void>
  onDecide: (arxivIds: string[], status: CandidateStatus) => Promise<void>
}) {
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState('relevance')
  const [filter, setFilter] = useState<Filter>('pending')
  const [error, setError] = useState<string | null>(null)

  const counts = useMemo(
    () => ({
      pending: candidates.filter((c) => c.status === 'pending').length,
      confirmed: candidates.filter((c) => c.status === 'confirmed').length,
      rejected: candidates.filter((c) => c.status === 'rejected').length,
    }),
    [candidates],
  )

  const shown = useMemo(
    () =>
      candidates
        .filter((c) => c.status === filter)
        .sort((a, b) => (b.decided_at ?? 0) - (a.decided_at ?? 0) || b.year - a.year),
    [candidates, filter],
  )

  const run = async () => {
    if (!query.trim() || busy) return
    setError(null)
    try {
      await onSearch(query.trim(), sortBy)
      setFilter('pending')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <section className="panel review">
      <div className="panel-head">
        <h3>Find papers</h3>
        {busy && <span className="spinner" />}
      </div>

      <div style={{ padding: '0 12px' }}>
        <div className="search-row" style={{ flexDirection: 'column', gap: 8 }}>
          <input
            className="field"
            placeholder="arXiv search — e.g. all:reranking RAG"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && run()}
          />
          <div style={{ display: 'flex', gap: 8 }}>
            <select
              className="field"
              style={{ flex: 1 }}
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="relevance">Most relevant</option>
              <option value="submittedDate">Newest first</option>
            </select>
            <button className="btn primary small" onClick={run} disabled={!query.trim() || busy}>
              Search
            </button>
          </div>
        </div>
        {error && <div className="turn-error" style={{ margin: '0 6px 10px' }}>{error}</div>}
        <div style={{ display: 'flex', gap: 4, padding: '4px 0 10px' }}>
          {(['pending', 'confirmed', 'rejected'] as Filter[]).map((f) => (
            <button
              key={f}
              className={`pill ${filter === f ? 'active' : ''}`}
              style={{ padding: '5px 11px', fontSize: 12.5 }}
              onClick={() => setFilter(f)}
            >
              {f} {counts[f]}
            </button>
          ))}
        </div>
      </div>

      <div className="panel-body">
        {!shown.length && (
          <div className="hint">
            {filter === 'pending'
              ? 'Nothing waiting. Search arXiv above — hits land here for you to confirm or reject.'
              : `No ${filter} papers yet.`}
          </div>
        )}

        {shown.map((c) => (
          <div key={c.arxiv_id} className="cand">
            <div className="c-title">{c.title}</div>
            <div className="c-meta">
              {c.authors.slice(0, 3).join(', ')}
              {c.authors.length > 3 ? ' et al.' : ''} · {c.year || 'n.d.'} · {c.arxiv_id}
            </div>
            <div className="c-abs">{c.abstract}</div>
            <div className="c-actions">
              {c.status === 'pending' ? (
                <>
                  <button
                    className="btn primary small"
                    disabled={busy}
                    onClick={() => onDecide([c.arxiv_id], 'confirmed')}
                  >
                    Confirm
                  </button>
                  <button
                    className="btn ghost small"
                    disabled={busy}
                    onClick={() => onDecide([c.arxiv_id], 'rejected')}
                  >
                    Reject
                  </button>
                </>
              ) : (
                <>
                  <span className={`status-tag ${c.status}`}>{c.status}</span>
                  <button
                    className="btn ghost small"
                    disabled={busy}
                    onClick={() =>
                      onDecide([c.arxiv_id], c.status === 'confirmed' ? 'rejected' : 'confirmed')
                    }
                  >
                    {c.status === 'confirmed' ? 'Remove' : 'Confirm'}
                  </button>
                </>
              )}
              <span className="spacer" />
              {c.url && (
                <a href={c.url} target="_blank" rel="noreferrer" style={{ fontSize: 12 }}>
                  arXiv ↗
                </a>
              )}
            </div>
          </div>
        ))}
      </div>

      {counts.pending > 1 && filter === 'pending' && (
        <div className="panel-foot">
          <button
            className="btn ghost small"
            style={{ width: '100%' }}
            disabled={busy}
            onClick={() =>
              onDecide(
                candidates.filter((c) => c.status === 'pending').map((c) => c.arxiv_id),
                'confirmed',
              )
            }
          >
            Confirm all {counts.pending} pending
          </button>
        </div>
      )}
    </section>
  )
}
