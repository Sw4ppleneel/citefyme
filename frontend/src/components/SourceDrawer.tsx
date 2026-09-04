import type { Source } from '../types'

/** Citation drill-down: read the cited chunk inside the whole source it came
 *  from, so a claim can be checked against its context rather than a snippet. */
export default function SourceDrawer({
  source,
  highlightChunkId,
  onClose,
}: {
  source: Source
  highlightChunkId?: string | null
  onClose: () => void
}) {
  return (
    <div className="drawer-scrim" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', gap: 12 }}>
          <div style={{ flex: 1 }}>
            <h3>{source.title}</h3>
            <div className="d-meta">
              {source.authors.slice(0, 4).join(', ')}
              {source.authors.length > 4 ? ' et al.' : ''} · {source.year || 'n.d.'} ·{' '}
              {source.chunks.length} chunks
            </div>
          </div>
          <button className="icon-btn" onClick={onClose}>
            ✕
          </button>
        </div>
        {source.chunks.map((c, i) => (
          <div key={c.chunk_id} className={`chunk ${c.chunk_id === highlightChunkId ? 'hit' : ''}`}>
            <div className="ch-head">
              {c.section} · chunk {i + 1}
              {c.chunk_id === highlightChunkId ? ' · cited here' : ''}
            </div>
            {c.text}
          </div>
        ))}
      </div>
    </div>
  )
}
