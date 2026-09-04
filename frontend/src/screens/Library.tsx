import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { NotebookSummary } from '../types'

const EMOJI = ['📒', '🔍', '🧬', '📊', '🧠', '⚗️', '📚', '🛰️', '🧮', '🩺', '⚖️', '🌍']
type Sort = 'recent' | 'title' | 'sources'
type Tab = 'all' | 'pending' | 'empty'

function relative(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

export default function Library({ onOpen }: { onOpen: (id: string) => void }) {
  const [books, setBooks] = useState<NotebookSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('all')
  const [query, setQuery] = useState('')
  const [view, setView] = useState<'grid' | 'list'>('grid')
  const [sort, setSort] = useState<Sort>('recent')
  const [creating, setCreating] = useState(false)
  const [menuFor, setMenuFor] = useState<string | null>(null)

  const load = () => {
    setLoading(true)
    api
      .listNotebooks()
      .then(setBooks)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }
  useEffect(load, [])

  const shown = useMemo(() => {
    let out = books
    if (tab === 'pending') out = out.filter((b) => b.pending_count > 0)
    else if (tab === 'empty') out = out.filter((b) => b.source_count === 0)
    const q = query.trim().toLowerCase()
    if (q) out = out.filter((b) => b.title.toLowerCase().includes(q))
    const sorted = [...out]
    if (sort === 'title') sorted.sort((a, b) => a.title.localeCompare(b.title))
    else if (sort === 'sources') sorted.sort((a, b) => b.source_count - a.source_count)
    else sorted.sort((a, b) => b.updated_at - a.updated_at)
    return sorted
  }, [books, tab, query, sort])

  const remove = async (nb: NotebookSummary) => {
    setMenuFor(null)
    if (!window.confirm(`Delete “${nb.title}” and its ${nb.source_count} sources?`)) return
    await api.deleteNotebook(nb.notebook_id)
    load()
  }

  const rename = async (nb: NotebookSummary) => {
    setMenuFor(null)
    const title = window.prompt('Notebook name', nb.title)
    if (!title?.trim()) return
    await api.updateNotebook(nb.notebook_id, { title: title.trim() })
    load()
  }

  return (
    <main className="library">
      <div className="library-inner">
        <div className="library-controls">
          <button className={`pill ${tab === 'all' ? 'active' : ''}`} onClick={() => setTab('all')}>
            All
          </button>
          <button
            className={`pill ${tab === 'pending' ? 'active' : ''}`}
            onClick={() => setTab('pending')}
          >
            Needs review
          </button>
          <button className={`pill ${tab === 'empty' ? 'active' : ''}`} onClick={() => setTab('empty')}>
            No sources yet
          </button>
          <span className="spacer" />
          <input
            className="field filter-search"
            placeholder="Search notebooks"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            className="pill"
            onClick={() => setView(view === 'grid' ? 'list' : 'grid')}
            title={view === 'grid' ? 'List view' : 'Grid view'}
          >
            {view === 'grid' ? '☰' : '▦'}
          </button>
          <select className="field" style={{ width: 'auto' }} value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
            <option value="recent">Most recent</option>
            <option value="title">Title</option>
            <option value="sources">Most sources</option>
          </select>
          <button className="btn primary" onClick={() => setCreating(true)}>
            + Create new
          </button>
        </div>

        <div className="section-head">
          <h2>Recent notebooks</h2>
          <span className="count">
            {books.length} notebook{books.length === 1 ? '' : 's'}
          </span>
        </div>

        {error && <div className="empty">Couldn’t load notebooks — {error}</div>}
        {loading && !books.length && <div className="hint">Loading…</div>}

        <div className={`grid ${view === 'list' ? 'list' : ''}`}>
          <button className="card-new" onClick={() => setCreating(true)}>
            <span className="plus">+</span>
            Create new notebook
          </button>

          {shown.map((nb) => (
            <div key={nb.notebook_id} className="card" onClick={() => onOpen(nb.notebook_id)} role="button">
              <span className="emoji">{nb.emoji}</span>
              <div className="card-title">{nb.title}</div>
              <div className="card-meta">
                {relative(nb.updated_at)} · {nb.source_count} source{nb.source_count === 1 ? '' : 's'}
                {nb.pending_count > 0 && ` · ${nb.pending_count} to review`}
              </div>
              <div className="card-menu menu-wrap" onClick={(e) => e.stopPropagation()}>
                <button
                  className="icon-btn"
                  onClick={() => setMenuFor(menuFor === nb.notebook_id ? null : nb.notebook_id)}
                >
                  ⋮
                </button>
                {menuFor === nb.notebook_id && (
                  <div className="menu">
                    <button onClick={() => rename(nb)}>Rename</button>
                    <button className="danger" onClick={() => remove(nb)}>
                      Delete
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {!loading && !shown.length && books.length > 0 && (
          <div className="empty" style={{ marginTop: 16 }}>
            No notebook matches this filter.
          </div>
        )}
      </div>

      {creating && (
        <CreateDialog
          onClose={() => setCreating(false)}
          onCreated={(id) => {
            setCreating(false)
            onOpen(id)
          }}
        />
      )}
    </main>
  )
}

function CreateDialog({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (id: string) => void
}) {
  const [title, setTitle] = useState('')
  const [emoji, setEmoji] = useState(EMOJI[0])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => inputRef.current?.focus(), [])

  const submit = async () => {
    if (!title.trim() || busy) return
    setBusy(true)
    try {
      const nb = await api.createNotebook(title.trim(), emoji)
      onCreated(nb.notebook_id)
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }

  return (
    <div className="drawer-scrim" onClick={onClose} style={{ alignItems: 'center', justifyContent: 'center' }}>
      <div
        className="drawer"
        style={{ height: 'auto', width: 'min(460px, 92vw)', borderRadius: 20 }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3>New notebook</h3>
        <div className="d-meta">One notebook per research topic.</div>
        <input
          ref={inputRef}
          className="field"
          placeholder="e.g. RAG techniques"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
        />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, margin: '16px 0 20px' }}>
          {EMOJI.map((e) => (
            <button
              key={e}
              className="pill"
              style={{
                fontSize: 20,
                padding: '6px 10px',
                background: e === emoji ? '#3b4c6b' : undefined,
              }}
              onClick={() => setEmoji(e)}
            >
              {e}
            </button>
          ))}
        </div>
        {error && <div className="turn-error" style={{ marginBottom: 14 }}>{error}</div>}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" disabled={!title.trim() || busy} onClick={submit}>
            {busy ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  )
}
