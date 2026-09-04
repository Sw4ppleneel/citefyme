import type { Source } from '../types'

export default function SourceList({
  sources,
  onOpen,
  onRemove,
  onAdd,
}: {
  sources: Source[]
  onOpen: (sourceId: string) => void
  onRemove: (sourceId: string) => void
  onAdd: () => void
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Sources</h3>
        <span className="badge">{sources.length}</span>
      </div>
      <div className="panel-body">
        {!sources.length && (
          <div className="hint">
            No confirmed sources yet. Search arXiv on the right, then confirm the papers that
            belong to this topic — answers are grounded only in what you confirm.
          </div>
        )}
        {sources.map((s) => (
          <div key={s.source_id} className="source-item">
            <span style={{ marginTop: 1 }}>📄</span>
            <button className="st" onClick={() => onOpen(s.source_id)} style={{ textAlign: 'left' }}>
              <div className="s-title">{s.title}</div>
              <div className="s-meta">
                {s.year || 'n.d.'} · {s.chunks.length} chunks
              </div>
            </button>
            <button
              className="icon-btn x"
              title="Remove from notebook"
              onClick={() => onRemove(s.source_id)}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
      <div className="panel-foot">
        <button className="btn ghost small" style={{ width: '100%' }} onClick={onAdd}>
          + Add sources
        </button>
      </div>
    </section>
  )
}
