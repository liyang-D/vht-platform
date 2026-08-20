import { useEffect, useMemo, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import "./App.css"
import ScenarioDialog, { type Scenario } from "./ScenarioDialog"
import "./ScenarioEnhancements.css"

type WorkspaceMode = 'live' | 'history' | 'replay'
type Side = 'left' | 'right'

type Adapter = {
  name: string
  loaded: boolean | null
  valid: boolean
  error: string | null
}

type Message = {
  id: string
  role: 'user' | 'assistant' | 'avatar'
  text: string
  created_at?: string
}

type Session = {
  id: string
  model_weights: string
  summary: string | null
  task_config?: { task_name?: string; role?: string; instructions?: string }
  created_at: string
  messages: Message[]
}

type HistoryResponse = {
  weights: string[]
  sessions: Session[]
}

type CreatedSide = {
  session_id: string
  model_weights: string
  opening_message: string
}

type PairResponse = {
  left: CreatedSide
  right: CreatedSide
}

type PairTurnResponse = {
  left: { text: string }
  right: { text: string }
}

type StoredSession = Omit<Session, 'id'> & { session_id: string }

type ReplayResponse = {
  source: Session
  target: StoredSession
  replayed_turn_count: number
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
  ?? import.meta.env.BASE_URL.replace(/\/$/, '')

function uid(prefix = 'message') {
  return typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

async function readError(response: Response) {
  const payload = await response.json().catch(() => null)
  return payload?.detail ?? `Request failed (${response.status})`
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)
  if (!response.ok) throw new Error(await readError(response))
  return response.json() as Promise<T>
}

function normalizeSession(session: StoredSession): Session {
  return { ...session, id: session.session_id }
}

function hasCompleteConversation(session: Session | null) {
  if (!session?.summary?.trim()) return false

  let hasUserTurn = false
  let waitingForResponse = false
  for (const message of session.messages) {
    if (message.role === "user") {
      if (waitingForResponse) return false
      hasUserTurn = true
      waitingForResponse = true
    } else if (waitingForResponse) {
      waitingForResponse = false
    }
  }
  return hasUserTurn && !waitingForResponse
}

function sessionOptionLabel(session: Session) {
  const turns = session.messages.filter((message) => message.role === 'user').length
  return `${session.model_weights} · ${turns} turn${turns === 1 ? '' : 's'} · ${formatDate(session.created_at)}`
}

function AdapterSelect({
  adapters,
  disabled,
  label,
  onChange,
  value,
}: {
  adapters: Adapter[]
  disabled?: boolean
  label: string
  onChange: (value: string) => void
  value: string
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select disabled={disabled} onChange={(event) => onChange(event.target.value)} value={value}>
        <option value="">Base model</option>
        {adapters.map((adapter) => (
          <option disabled={!adapter.valid} key={adapter.name} value={adapter.name}>
            {adapter.name}{adapter.loaded ? ' · loaded' : ''}{adapter.valid ? '' : ' · unavailable'}
          </option>
        ))}
      </select>
    </label>
  )
}

function SessionSelect({
  label,
  onChange,
  sessions,
  value,
}: {
  label: string
  onChange: (value: string) => void
  sessions: Session[]
  value: string
}) {
  return (
    <label className="field history-select">
      <span>{label}</span>
      <select onChange={(event) => onChange(event.target.value)} value={value}>
        <option value="">Choose a recorded session</option>
        {sessions.map((session) => (
          <option key={session.id} value={session.id}>{sessionOptionLabel(session)}</option>
        ))}
      </select>
    </label>
  )
}

function ModelPane({
  emptyText,
  isPending,
  session,
  side,
}: {
  emptyText: string
  isPending: boolean
  session: Session | null
  side: Side
}) {
  return (
    <section className={`model-pane ${side}`}>
      <header className="pane-header">
        <div className="side-mark">{side === 'left' ? 'A' : 'B'}</div>
        <div className="pane-identity">
          <span>Model weights</span>
          <strong>{session?.model_weights ?? 'Not selected'}</strong>
        </div>
        {session && <code title={session.id}>{session.id.slice(0, 8)}</code>}
      </header>

      {session?.task_config?.role && (
        <details className="pane-scenario">
          <summary>View session scenario</summary>
          <div>
            <span>Role</span>
            <p>{session.task_config.role}</p>
            <span>Instructions</span>
            <p>{session.task_config.instructions}</p>
          </div>
        </details>
      )}

      <div className="transcript">
        {!session && !isPending && <div className="pane-empty">{emptyText}</div>}
        {session?.messages.map((message) => {
          const role = message.role === 'user' ? 'user' : 'assistant'
          return (
            <article className={`turn ${role}`} key={message.id}>
              <span>{role === 'user' ? 'User' : 'Model'}</span>
              <p>{message.text}</p>
            </article>
          )
        })}
        {isPending && (
          <article className="turn assistant pending-turn">
            <span>Model</span>
            <p><i /><i /><i /></p>
          </article>
        )}
        {session?.summary && (
          <aside className="session-summary">
            <span>Archived summary</span>
            <p>{session.summary}</p>
          </aside>
        )}
      </div>
    </section>
  )
}

function ModeButton({ active, children, onClick }: {
  active: boolean
  children: ReactNode
  onClick: () => void
}) {
  return <button className={active ? 'active' : ''} onClick={onClick} type="button">{children}</button>
}

function App() {
  const [mode, setMode] = useState<WorkspaceMode>('live')
  const [adapters, setAdapters] = useState<Adapter[]>([])
  const [history, setHistory] = useState<HistoryResponse>({ weights: [], sessions: [] })
  const [leftAdapter, setLeftAdapter] = useState('')
  const [rightAdapter, setRightAdapter] = useState('')
  const [leftLive, setLeftLive] = useState<Session | null>(null)
  const [rightLive, setRightLive] = useState<Session | null>(null)
  const [liveActive, setLiveActive] = useState(false)
  const [historyLeftId, setHistoryLeftId] = useState('')
  const [historyRightId, setHistoryRightId] = useState('')
  const [replaySourceId, setReplaySourceId] = useState('')
  const [replaySourceSide, setReplaySourceSide] = useState<Side>('left')
  const [replayAdapter, setReplayAdapter] = useState('')
  const [replaySource, setReplaySource] = useState<Session | null>(null)
  const [replayTarget, setReplayTarget] = useState<Session | null>(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [status, setStatus] = useState('Loading model weights and recorded sessions…')
  const [defaultScenario, setDefaultScenario] = useState<Scenario | null>(null)
  const [scenarioMode, setScenarioMode] = useState<"default" | "custom">("default")
  const [customRole, setCustomRole] = useState("")
  const [customInstructions, setCustomInstructions] = useState("")
  const [isScenarioOpen, setIsScenarioOpen] = useState(false)

  async function loadResources() {
    try {
      const [adapterPayload, historyPayload, scenarioPayload] = await Promise.all([
        fetchJson<{ adapters: Adapter[] }>("/api/loras"),
        fetchJson<HistoryResponse>("/api/history"),
        fetchJson<{ default: Scenario }>("/api/scenario"),
      ])
      setAdapters(adapterPayload.adapters)
      setHistory(historyPayload)
      setDefaultScenario(scenarioPayload.default)
      setCustomRole((current) => current || scenarioPayload.default.role)
      setCustomInstructions((current) => current || scenarioPayload.default.instructions)
      setHistoryLeftId((current) => current || historyPayload.sessions[0]?.id || '')
      setHistoryRightId((current) => current || historyPayload.sessions[1]?.id || historyPayload.sessions[0]?.id || '')
      setReplaySourceId((current) => current || historyPayload.sessions[0]?.id || '')
      setStatus(`${historyPayload.sessions.length} traceable sessions available`)
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not load evaluation workspace')
    }
  }

  useEffect(() => {
    async function loadInitialResources() {
      await loadResources()
    }

    void loadInitialResources()
  }, [])

  const historyLeft = useMemo(
    () => history.sessions.find((session) => session.id === historyLeftId) ?? null,
    [history.sessions, historyLeftId],
  )
  const historyRight = useMemo(
    () => history.sessions.find((session) => session.id === historyRightId) ?? null,
    [history.sessions, historyRightId],
  )
  const selectedReplaySource = useMemo(
    () => history.sessions.find((session) => session.id === replaySourceId) ?? null,
    [history.sessions, replaySourceId],
  )

  const configuredScenario: Scenario | null = scenarioMode === "custom"
    ? { name: "Custom scenario", role: customRole, instructions: customInstructions }
    : defaultScenario

  const displayedLeft = mode === "live"
    ? leftLive
    : mode === 'history'
      ? historyLeft
      : replaySourceSide === 'left' ? (replaySource ?? selectedReplaySource) : replayTarget
  const displayedRight = mode === "live"
    ? rightLive
    : mode === "history"
      ? historyRight
      : replaySourceSide === "right" ? (replaySource ?? selectedReplaySource) : replayTarget
  const canExport = Boolean(
    displayedLeft
    && displayedRight
    && displayedLeft.id !== displayedRight.id
    && hasCompleteConversation(displayedLeft)
    && hasCompleteConversation(displayedRight)
  )
  const exportHint = !displayedLeft || !displayedRight
    ? "Select or create both comparison sessions first"
    : displayedLeft.id === displayedRight.id
      ? "Choose two different sessions"
      : !hasCompleteConversation(displayedLeft) || !hasCompleteConversation(displayedRight)
        ? "Both sessions must be ended with complete user/model turns"
        : "Export both displayed sessions as one traceable JSON comparison"

  async function startPair() {
    setBusy(true)
    setStatus('Creating model A, then model B…')
    try {
      const payload = await fetchJson<PairResponse>('/api/pairs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          left_adapter: leftAdapter || null,
          right_adapter: rightAdapter || null,
          scenario: scenarioMode === "custom"
            ? { role: customRole, instructions: customInstructions }
            : null,
        }),
      })
      const now = new Date().toISOString()
      setLeftLive({
        id: payload.left.session_id,
        model_weights: payload.left.model_weights,
        summary: null,
        task_config: configuredScenario
          ? { role: configuredScenario.role, instructions: configuredScenario.instructions }
          : undefined,
        created_at: now,
        messages: [{ id: uid(), role: 'assistant', text: payload.left.opening_message }],
      })
      setRightLive({
        id: payload.right.session_id,
        model_weights: payload.right.model_weights,
        summary: null,
        task_config: configuredScenario
          ? { role: configuredScenario.role, instructions: configuredScenario.instructions }
          : undefined,
        created_at: now,
        messages: [{ id: uid(), role: 'assistant', text: payload.right.opening_message }],
      })
      setLiveActive(true)
      setStatus('Paired sessions ready · turns run A then B')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not start paired sessions')
    } finally {
      setBusy(false)
    }
  }

  async function sendTurn(event: FormEvent) {
    event.preventDefault()
    const text = input.trim()
    if (!text || !leftLive || !rightLive || !liveActive || busy) return

    const userMessage = { id: uid(), role: 'user' as const, text }
    setLeftLive((session) => session && ({ ...session, messages: [...session.messages, userMessage] }))
    setRightLive((session) => session && ({ ...session, messages: [...session.messages, { ...userMessage, id: uid() }] }))
    setInput('')
    setBusy(true)
    setStatus('Running model A, then model B…')
    try {
      const payload = await fetchJson<PairTurnResponse>('/api/pairs/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          left_session_id: leftLive.id,
          right_session_id: rightLive.id,
          text,
        }),
      })
      setLeftLive((session) => session && ({
        ...session,
        messages: [...session.messages, { id: uid(), role: 'assistant', text: payload.left.text }],
      }))
      setRightLive((session) => session && ({
        ...session,
        messages: [...session.messages, { id: uid(), role: 'assistant', text: payload.right.text }],
      }))
      setStatus('Both responses complete')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not run paired turn')
    } finally {
      setBusy(false)
    }
  }

  async function endPair() {
    if (!leftLive || !rightLive || busy) return
    setBusy(true)
    setStatus('Archiving model A, then model B…')
    try {
      const payload = await fetchJson<{ left: { summary: string }, right: { summary: string } }>('/api/pairs/end', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ left_session_id: leftLive.id, right_session_id: rightLive.id }),
      })
      setLeftLive((session) => session && ({ ...session, summary: payload.left.summary }))
      setRightLive((session) => session && ({ ...session, summary: payload.right.summary }))
      setLiveActive(false)
      await loadResources()
      setStatus('Both sessions archived independently')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not archive paired sessions')
    } finally {
      setBusy(false)
    }
  }

  async function runReplay() {
    if (!selectedReplaySource || busy) return
    setBusy(true)
    setReplaySource(selectedReplaySource)
    setReplayTarget(null)
    const turnCount = selectedReplaySource.messages.filter((message) => message.role === 'user').length
    setStatus(`Replaying ${turnCount} user turns in order…`)
    try {
      const payload = await fetchJson<ReplayResponse>('/api/replays', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_session_id: selectedReplaySource.id,
          target_adapter: replayAdapter || null,
        }),
      })
      setReplaySource(payload.source)
      setReplayTarget(normalizeSession(payload.target))
      await loadResources()
      setStatus(`${payload.replayed_turn_count} user turns replayed and archived`)
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not replay session')
    } finally {
      setBusy(false)
    }
  }

  async function exportComparison() {
    if (!displayedLeft || !displayedRight || !canExport || exporting) return

    const leftSession = displayedLeft
    const rightSession = displayedRight
    setExporting(true)
    setStatus("Preparing verified comparison export…")
    try {
      const response = await fetch(API_BASE_URL + "/api/exports", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_session_id: leftSession.id,
          right_session_id: rightSession.id,
          mode,
        }),
      })
      if (!response.ok) throw new Error(await readError(response))

      const contentDisposition = response.headers.get("Content-Disposition") ?? ""
      const filename = contentDisposition.match(/filename="?([^";]+)"?/)?.[1]
        ?? "evaluation-comparison-" + new Date().toISOString().replace(/[:.]/g, "-") + ".json"
      const objectUrl = URL.createObjectURL(await response.blob())
      const link = document.createElement("a")
      link.href = objectUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0)
      setStatus("Exported " + leftSession.model_weights + " vs " + rightSession.model_weights)
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not export comparison")
    } finally {
      setExporting(false)
    }
  }

  function switchMode(nextMode: WorkspaceMode) {
    setMode(nextMode)
    setStatus(nextMode === 'live'
      ? 'Create two independent sessions for a shared live conversation'
      : nextMode === 'history'
        ? 'Choose any two recorded sessions to inspect side by side'
        : 'Keep one recorded session and replay its user turns against new weights')
  }

  return (
    <main className="evaluation-shell">
      <section className="workspace">
        <header className="product-header">
          <div className="product-lockup">
            <div className="product-icon">E</div>
            <div>
              <div className="eyebrow">Experimental workspace</div>
              <h1>Model Evaluation</h1>
              <p>{status}</p>
            </div>
          </div>
          <div className="product-actions">
            <button
              className="scenario-header-button export-header-button"
              disabled={!canExport || exporting}
              onClick={() => void exportComparison()}
              title={exportHint}
              type="button"
            >
              {exporting ? "Exporting…" : "Export comparison"}
            </button>
            <button className="scenario-header-button" disabled={!defaultScenario} onClick={() => setIsScenarioOpen(true)} type="button">Scenario</button>
            <a className="simple-chat-link" href="/chat/">Open Simple Chat <span>↗</span></a>
          </div>
        </header>

        <nav className="mode-tabs" aria-label="Evaluation mode">
          <ModeButton active={mode === 'live'} onClick={() => switchMode('live')}>Live A/B</ModeButton>
          <ModeButton active={mode === 'history'} onClick={() => switchMode('history')}>History compare</ModeButton>
          <ModeButton active={mode === 'replay'} onClick={() => switchMode('replay')}>Replay session</ModeButton>
        </nav>

        <section className="configuration">
          {mode === 'live' && (
            <>
              <AdapterSelect adapters={adapters} disabled={liveActive || busy} label="Model A weights" onChange={setLeftAdapter} value={leftAdapter} />
              <div className="config-action">
                <span>One prompt · two independent sessions</span>
                {!liveActive
                  ? <button disabled={busy || (scenarioMode === "custom" && (!customRole.trim() || !customInstructions.trim()))} onClick={startPair} type="button">Start paired chat</button>
                  : <button className="secondary" disabled={busy} onClick={endPair} type="button">End & archive both</button>}
              </div>
              <AdapterSelect adapters={adapters} disabled={liveActive || busy} label="Model B weights" onChange={setRightAdapter} value={rightAdapter} />
            </>
          )}

          {mode === 'history' && (
            <>
              <SessionSelect label="Session A" onChange={setHistoryLeftId} sessions={history.sessions} value={historyLeftId} />
              <div className="config-action quiet"><span>Weights shown below come from the database</span></div>
              <SessionSelect label="Session B" onChange={setHistoryRightId} sessions={history.sessions} value={historyRightId} />
            </>
          )}

          {mode === 'replay' && (
            <>
              <SessionSelect label="Recorded source" onChange={setReplaySourceId} sessions={history.sessions} value={replaySourceId} />
              <div className="replay-controls">
                <div className="side-toggle" aria-label="Recorded session position">
                  <button className={replaySourceSide === 'left' ? 'active' : ''} onClick={() => setReplaySourceSide('left')} type="button">Source on A</button>
                  <button className={replaySourceSide === 'right' ? 'active' : ''} onClick={() => setReplaySourceSide('right')} type="button">Source on B</button>
                </div>
                <button disabled={!replaySourceId || busy} onClick={runReplay} type="button">Replay user turns</button>
              </div>
              <AdapterSelect adapters={adapters} disabled={busy} label="New comparison weights" onChange={setReplayAdapter} value={replayAdapter} />
            </>
          )}
        </section>

        <div className="comparison">
          <ModelPane
            emptyText={mode === 'live' ? 'Choose weights and start a paired chat.' : 'Choose a session for model A.'}
            isPending={busy && (mode === 'live' || (mode === 'replay' && replaySourceSide === 'right'))}
            session={displayedLeft}
            side="left"
          />
          <div className="comparison-divider"><span>VS</span></div>
          <ModelPane
            emptyText={mode === 'live' ? 'The same user turns will appear here.' : 'Choose a session for model B.'}
            isPending={busy && (mode === 'live' || (mode === 'replay' && replaySourceSide === 'left'))}
            session={displayedRight}
            side="right"
          />
        </div>

        {mode === 'live' && (
          <form className="shared-composer" onSubmit={sendTurn}>
            <div className="composer-caption">
              <span>Shared user turn</span>
              <small>Submitted once, processed A → B</small>
            </div>
            <textarea
              disabled={!liveActive || busy}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }}
              placeholder={liveActive ? 'Type the next turn for both models…' : 'Start a paired chat to enter turns'}
              rows={1}
              value={input}
            />
            <button disabled={!liveActive || busy || !input.trim()} type="submit">Run both</button>
          </form>
        )}
      </section>
      <ScenarioDialog
        customInstructions={customInstructions}
        customRole={customRole}
        defaultScenario={defaultScenario}
        isLocked={liveActive}
        isOpen={isScenarioOpen}
        mode={scenarioMode}
        onClose={() => setIsScenarioOpen(false)}
        onCustomInstructionsChange={setCustomInstructions}
        onCustomRoleChange={setCustomRole}
        onModeChange={setScenarioMode}
      />
    </main>
  )
}

export default App
