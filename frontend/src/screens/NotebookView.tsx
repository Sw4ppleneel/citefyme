import { useCallback, useEffect, useState } from 'react'
import Chat from '../components/Chat'
import SearchPanel from '../components/SearchPanel'
import SourceDrawer from '../components/SourceDrawer'
import SourceList from '../components/SourceList'
import { api } from '../api'
import type { CandidateStatus, NotebookDetail, RetrievalMode, Source } from '../types'

export default function NotebookView({
  notebookId,
  onBack,
}: {
  notebookId: string
  onBack: () => void
}) {
  const [nb, setNb] = useState<NotebookDetail | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [askError, setAskError] = useState<string | null>(null)
  const [asking, setAsking] = useState(false)
  const [searchBusy, setSearchBusy] = useState(false)
  // dense alone beat hybrid RRF and matched hybrid+rerank on the benchmark
  // corpus, so it is the default and rerank starts off
  const [mode, setMode] = useState<RetrievalMode>('dense')
  const [rerank, setRerank] = useState(false)
  const [drawer, setDrawer] = useState<{ source: Source; chunkId: string | null } | null>(null)

  const load = useCallback(() => {
    api
      .getNotebook(notebookId)
      .then(setNb)
      .catch((e: Error) => setLoadError(e.message))
  }, [notebookId])

  useEffect(load, [load])

  if (loadError) {
    return (
      <main className="library">
        <div className="library-inner">
          <div className="empty">
            Couldn’t open this notebook — {loadError}
            <div style={{ marginTop: 16 }}>
              <button className="btn ghost" onClick={onBack}>
                ← Back to notebooks
              </button>
            </div>
          </div>
        </div>
      </main>
    )
  }
  if (!nb) return <div className="hint" style={{ padding: 32 }}>Loading notebook…</div>

  const ask = async (question: string) => {
    setAsking(true)
    setAskError(null)
    try {
      await api.ask(notebookId, question, mode, rerank)
    } catch (e) {
      setAskError((e as Error).message)
    } finally {
      // the turn is persisted server-side either way (failures included), so a
      // reload is what shows it — including the error turn
      setAsking(false)
      load()
    }
  }

  const search = async (query: string, sortBy: string) => {
    setSearchBusy(true)
    try {
      await api.search(notebookId, query, 12, sortBy)
      const fresh = await api.getNotebook(notebookId)
      setNb(fresh)
    } finally {
      setSearchBusy(false)
    }
  }

  const decide = async (arxivIds: string[], status: CandidateStatus) => {
    setSearchBusy(true)
    try {
      await api.decide(notebookId, arxivIds, status)
      const fresh = await api.getNotebook(notebookId)
      setNb(fresh)
    } finally {
      setSearchBusy(false)
    }
  }

  const openSource = async (sourceId: string, chunkId: string | null = null) => {
    const found = nb.sources.find((s) => s.source_id === sourceId)
    if (found) {
      setDrawer({ source: found, chunkId })
      return
    }
    // cited source was removed from the notebook since the answer was written
    try {
      const { source } = await api.sourceChunks(notebookId, sourceId)
      setDrawer({ source, chunkId })
    } catch {
      setAskError(`That citation points at a source no longer in this notebook (${sourceId}).`)
    }
  }

  const rename = async () => {
    const title = window.prompt('Notebook name', nb.title)
    if (!title?.trim()) return
    await api.updateNotebook(notebookId, { title: title.trim() })
    load()
  }

  const removeSource = async (sourceId: string) => {
    await api.removeSource(notebookId, sourceId)
    const fresh = await api.getNotebook(notebookId)
    setNb(fresh)
  }

  const focusSearch = () => {
    document.querySelector<HTMLInputElement>('.panel.review input.field')?.focus()
  }

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '0 24px 12px' }}>
        <button className="icon-btn" onClick={onBack} title="All notebooks">
          ←
        </button>
        <span style={{ fontSize: 20 }}>{nb.emoji}</span>
        <button onClick={rename} style={{ fontSize: 17, fontWeight: 500 }} title="Rename">
          {nb.title}
        </button>
        <span style={{ color: 'var(--text-faint)', fontSize: 12.5 }}>
          {nb.source_count} source{nb.source_count === 1 ? '' : 's'} · {nb.chunk_count} chunks
          {nb.pending_count > 0 && ` · ${nb.pending_count} to review`}
        </span>
      </div>

      <div className="workspace">
        <SourceList
          sources={nb.sources}
          onOpen={(id) => openSource(id)}
          onRemove={removeSource}
          onAdd={focusSearch}
        />
        <Chat
          title={nb.title}
          emoji={nb.emoji}
          sources={nb.sources}
          turns={nb.chat}
          busy={asking}
          mode={mode}
          rerank={rerank}
          onModeChange={setMode}
          onRerankChange={setRerank}
          onAsk={ask}
          onCite={(sourceId, chunkId) => openSource(sourceId, chunkId)}
          error={askError}
        />
        <SearchPanel
          candidates={nb.candidates}
          busy={searchBusy}
          onSearch={search}
          onDecide={decide}
        />
      </div>

      {drawer && (
        <SourceDrawer
          source={drawer.source}
          highlightChunkId={drawer.chunkId}
          onClose={() => setDrawer(null)}
        />
      )}
    </>
  )
}
