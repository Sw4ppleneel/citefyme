import { useEffect, useRef, useState } from 'react'
import type { ChatTurn, Claim, RetrievalMode, Source } from '../types'

/** Two chips citing different chunks of the same paper would otherwise render
 *  identically (same title, and every arXiv source has one 'Abstract' section),
 *  so the chunk ordinal is what tells them apart. */
function chunkOrdinal(chunkId: string): string {
  const m = chunkId.match(/_c(\d+)$/)
  return m ? `chunk ${Number(m[1]) + 1}` : chunkId
}

const STATE_LABEL: Record<string, string> = {
  supported: 'supported by the cited passage',
  partial: 'only partly supported',
  unsupported: 'not supported by any retrieved passage',
  conflicting: 'evidence conflicts',
}

function ClaimRow({
  claim,
  sourceTitles,
  onCite,
}: {
  claim: Claim
  sourceTitles: Record<string, string>
  onCite: (sourceId: string, chunkId: string) => void
}) {
  const state = claim.citation_state ?? 'unsupported'
  return (
    <div className="claim">
      <span className={`state ${state}`} title={STATE_LABEL[state]} />
      <div className="ct">
        <div className="claim-text">{claim.text}</div>
        {claim.evidence.length > 0 && (
          <div className="cites">
            {claim.evidence.map((ev, i) => (
              <button
                key={ev.evidence_id}
                className="cite"
                title={`${ev.section} · ${ev.text}`}
                onClick={() => onCite(ev.source_id, ev.chunk_id)}
              >
                <span className="n">{i + 1}</span>
                <span className="cite-label">{sourceTitles[ev.source_id] ?? ev.source_id}</span>
                <span className="cite-chunk">{chunkOrdinal(ev.chunk_id)}</span>
              </button>
            ))}
          </div>
        )}
        {claim.verification_note && <div className="note">{claim.verification_note}</div>}
      </div>
    </div>
  )
}

export default function Chat({
  title,
  emoji,
  sources,
  turns,
  busy,
  mode,
  rerank,
  onModeChange,
  onRerankChange,
  onAsk,
  onCite,
  error,
}: {
  title: string
  emoji: string
  sources: Source[]
  turns: ChatTurn[]
  busy: boolean
  mode: RetrievalMode
  rerank: boolean
  onModeChange: (m: RetrievalMode) => void
  onRerankChange: (v: boolean) => void
  onAsk: (question: string) => void
  onCite: (sourceId: string, chunkId: string) => void
  error: string | null
}) {
  const [draft, setDraft] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const sourceTitles: Record<string, string> = {}
  for (const s of sources) sourceTitles[s.source_id] = s.title

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [turns.length, busy])

  const send = () => {
    const q = draft.trim()
    if (!q || busy || !sources.length) return
    onAsk(q)
    setDraft('')
  }

  return (
    <section className="panel" style={{ background: 'transparent' }}>
      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-inner">
          <div className="notebook-hero">
            <div className="emoji">{emoji}</div>
            <h1>{title}</h1>
            <p>
              {sources.length
                ? `Grounded in ${sources.length} confirmed paper${sources.length === 1 ? '' : 's'}. Every claim below carries the passage it came from.`
                : 'Confirm some papers first — this notebook answers only from sources you have accepted.'}
            </p>
          </div>

          {turns.map((t) => (
            <div key={t.turn_id} className="turn">
              <div className="q-row">
                <div className="q">{t.question}</div>
              </div>
              <div className="a">
                {t.error ? (
                  <div className="turn-error">{t.error}</div>
                ) : (
                  t.claims.map((c) => (
                    <ClaimRow key={c.claim_id} claim={c} sourceTitles={sourceTitles} onCite={onCite} />
                  ))
                )}
                <div className="turn-meta">
                  {t.mode}
                  {t.rerank ? ' + rerank' : ''} · {(t.latency_ms / 1000).toFixed(1)}s
                  {t.claims.length
                    ? ` · ${t.claims.filter((c) => c.citation_state === 'supported').length}/${t.claims.length} claims fully supported`
                    : ''}
                  {t.trace_id ? ` · trace ${t.trace_id}` : ''}
                </div>
              </div>
            </div>
          ))}

          {busy && (
            <div className="turn" style={{ display: 'flex', gap: 10, alignItems: 'center', color: 'var(--text-dim)' }}>
              <span className="spinner" />
              Retrieving, extracting claims, verifying each one against its passage…
            </div>
          )}
          {error && !busy && <div className="turn-error">{error}</div>}
        </div>
      </div>

      <div className="composer">
        <div className="composer-inner">
          <div className="composer-box">
            <textarea
              rows={1}
              placeholder={sources.length ? 'Ask about your sources…' : 'Confirm a source to start asking'}
              value={draft}
              disabled={!sources.length}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
            />
            <button className="send" onClick={send} disabled={!draft.trim() || busy || !sources.length}>
              ↑
            </button>
          </div>
          <div className="composer-opts">
            <label className="toggle">
              retrieval
              <select value={mode} onChange={(e) => onModeChange(e.target.value as RetrievalMode)}>
                <option value="dense">dense (best on this corpus)</option>
                <option value="bm25">bm25</option>
                <option value="hybrid">hybrid RRF</option>
              </select>
            </label>
            <label className="toggle">
              <input type="checkbox" checked={rerank} onChange={(e) => onRerankChange(e.target.checked)} />
              LLM rerank
            </label>
            <span style={{ color: 'var(--text-faint)' }}>
              Enter to send · Shift+Enter for a newline
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}
