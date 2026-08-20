import { useEffect, useMemo, useState } from 'react'
import './App.css'
import './SimpleChat.css'

type HistoryMessage = {
  id: string
  role: 'user' | 'avatar' | 'assistant'
  text: string
  created_at: string
}

type HistorySession = {
  id: string
  model_weights: string
  summary: string | null
  task_config: {
    task_name?: string
  } | null
  created_at: string
  updated_at: string
  messages: HistoryMessage[]
}

type HistoryResponse = {
  weights: string[]
  sessions: HistorySession[]
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
  ?? import.meta.env.BASE_URL.replace(/\/$/, '')
const CHAT_URL = import.meta.env.BASE_URL
const EVALUATION_URL = '/evaluation/'

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function HistoryPage() {
  const [history, setHistory] = useState<HistoryResponse>({ weights: [], sessions: [] })
  const [selectedWeights, setSelectedWeights] = useState<string[]>([])
  const [status, setStatus] = useState('Loading recorded sessions...')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let ignore = false

    async function loadHistory() {
      setIsLoading(true)
      setStatus('Loading recorded sessions...')

      try {
        const search = new URLSearchParams()
        selectedWeights.forEach((weights) => search.append('model_weights', weights))
        const query = search.size ? `?${search.toString()}` : ''
        const response = await fetch(`${API_BASE_URL}/api/history${query}`)

        if (!response.ok) {
          const payload = await response.json().catch(() => null)
          throw new Error(payload?.detail ?? 'Could not load session history')
        }

        const payload = (await response.json()) as HistoryResponse
        if (!ignore) {
          setHistory(payload)
          setStatus(`${payload.sessions.length} recorded session${payload.sessions.length === 1 ? '' : 's'}`)
        }
      } catch (error) {
        if (!ignore) {
          setStatus(error instanceof Error ? error.message : 'Could not load session history')
        }
      } finally {
        if (!ignore) {
          setIsLoading(false)
        }
      }
    }

    void loadHistory()
    return () => {
      ignore = true
    }
  }, [selectedWeights])

  const comparisonWeights = useMemo(
    () => selectedWeights.length ? selectedWeights : history.weights,
    [history.weights, selectedWeights],
  )

  function toggleWeight(weights: string) {
    setSelectedWeights((current) => current.includes(weights)
      ? current.filter((value) => value !== weights)
      : [...current, weights])
  }

  return (
    <main className="app-shell history-shell">
      <section className="history-panel" aria-label="Simple Chat session history">
        <header className="chat-header history-header">
          <div>
            <div className="title-line">
              <h1>Simple Chat</h1>
            </div>
            <p>{status}</p>
          </div>
          <nav className="header-actions" aria-label="Evaluation navigation">
            <a className="nav-button secondary" href={CHAT_URL}>New chat</a>
            <a className="nav-button secondary" href={EVALUATION_URL}>Evaluation</a>
            <span className="nav-button active" aria-current="page">Session history</span>
          </nav>
        </header>

        <section className="history-toolbar" aria-label="Model weights filters">
          <div>
            <h2>Compare model weights</h2>
            <p>Select one or more weights. Sessions without recorded weights are excluded.</p>
          </div>
          <button
            className={selectedWeights.length === 0 ? 'filter-chip active' : 'filter-chip'}
            onClick={() => setSelectedWeights([])}
            type="button"
          >
            All weights
          </button>
          {history.weights.map((weights) => (
            <button
              className={selectedWeights.includes(weights) ? 'filter-chip active' : 'filter-chip'}
              key={weights}
              onClick={() => toggleWeight(weights)}
              type="button"
            >
              {weights}
            </button>
          ))}
        </section>

        <div className="history-content">
          {!isLoading && history.weights.length === 0 && (
            <div className="empty-state history-empty">
              <h2>No traceable sessions yet</h2>
              <p>
                New chats will appear here after they record model weights.
                Legacy sessions are intentionally not included.
              </p>
              <a className="nav-button active" href={CHAT_URL}>Start a chat</a>
            </div>
          )}

          {comparisonWeights.length > 0 && (
            <div className="comparison-grid">
              {comparisonWeights.map((weights) => {
                const sessions = history.sessions.filter(
                  (session) => session.model_weights === weights,
                )

                return (
                  <section className="history-lane" key={weights}>
                    <header className="lane-header">
                      <div>
                        <span>Model weights</span>
                        <h2>{weights}</h2>
                      </div>
                      <strong>{sessions.length}</strong>
                    </header>

                    <div className="session-card-list">
                      {sessions.length === 0 && (
                        <p className="lane-empty">No sessions match this filter.</p>
                      )}
                      {sessions.map((session) => (
                        <details className="history-card" key={session.id}>
                          <summary>
                            <div>
                              <strong>{session.task_config?.task_name ?? 'Chat session'}</strong>
                              <span>{formatDate(session.created_at)}</span>
                            </div>
                            <span className={session.summary ? 'status-dot completed' : 'status-dot'}>
                              {session.summary ? 'Completed' : 'Active'}
                            </span>
                          </summary>

                          <div className="history-card-body">
                            {session.summary && (
                              <section className="history-summary">
                                <h3>Summary</h3>
                                <p>{session.summary}</p>
                              </section>
                            )}
                            <section className="history-transcript">
                              <h3>Transcript · {session.messages.length} messages</h3>
                              {session.messages.map((message) => (
                                <article className={`history-message ${message.role}`} key={message.id}>
                                  <span>{message.role === 'user' ? 'Participant' : 'Assistant'}</span>
                                  <p>{message.text}</p>
                                </article>
                              ))}
                            </section>
                          </div>
                        </details>
                      ))}
                    </div>
                  </section>
                )
              })}
            </div>
          )}
        </div>
      </section>
    </main>
  )
}

export default HistoryPage
