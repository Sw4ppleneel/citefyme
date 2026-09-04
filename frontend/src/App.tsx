import { useEffect, useState } from 'react'
import Library from './screens/Library'
import NotebookView from './screens/NotebookView'
import { api } from './api'
import type { Health } from './types'

/** Hash routing, so a refresh keeps you in the notebook you were reading.
 *  '#/' -> library, '#/n/<id>' -> one notebook. */
function routeFromHash(): string | null {
  const m = window.location.hash.match(/^#\/n\/(.+)$/)
  return m ? m[1] : null
}

export default function App() {
  const [notebookId, setNotebookId] = useState<string | null>(routeFromHash)
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  useEffect(() => {
    const onHash = () => setNotebookId(routeFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    api.health().then(setHealth).catch((e: Error) => setHealthError(e.message))
  }, [])

  const open = (id: string) => {
    window.location.hash = `#/n/${id}`
    setNotebookId(id)
  }
  const goHome = () => {
    window.location.hash = '#/'
    setNotebookId(null)
  }

  return (
    <div className="app">
      <header className="topbar">
        <button className="brand" onClick={goHome}>
          <span className="brand-mark">◆</span>
          CitefyMe
        </button>
        <span className="topbar-spacer" />
        {healthError ? (
          <span className="health-pill">
            <span className="dot bad" /> backend unreachable
          </span>
        ) : health ? (
          <span className="health-pill" title={`reranker: ${health.reranker}`}>
            <span className={`dot ${health.live_models ? '' : 'warn'}`} />
            {health.live_models ? health.llm.model : 'offline fallbacks'}
          </span>
        ) : null}
      </header>

      {healthError && (
        <div className="banner">
          Can’t reach the API on port 8002 — start it with{' '}
          <code>.venv/bin/uvicorn api.main:app --reload --port 8002</code> ({healthError})
        </div>
      )}
      {health && !health.live_models && (
        <div className="banner">
          No <code>GEMINI_API_KEY</code> found, so retrieval and answers come from the offline
          stub providers. Numbers and citations here are code sanity checks, not real quality.
        </div>
      )}

      {notebookId ? (
        <NotebookView notebookId={notebookId} onBack={goHome} />
      ) : (
        <Library onOpen={open} />
      )}
    </div>
  )
}
