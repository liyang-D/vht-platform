import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  text: string
}

type CreateSessionResponse = {
  opening_message: string
}

type StoredSessionMessage = {
  id: string
  role: 'user' | 'avatar' | 'assistant'
  text: string
}

type GetSessionResponse = {
  summary: string | null
  messages: StoredSessionMessage[]
}

type SendMessageResponse = {
  text: string
}

type EndSessionResponse = {
  summary: string
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8010'

function messageId() {
  return crypto.randomUUID()
}

function toChatMessage(message: StoredSessionMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role === 'user' ? 'user' : 'assistant',
    text: message.text,
  }
}

async function readError(response: Response) {
  try {
    const payload = await response.json()
    return payload.detail ?? 'Request failed'
  } catch {
    return 'Request failed'
  }
}

function App() {
  const [hasSession, setHasSession] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [summary, setSummary] = useState<string | null>(null)
  const [status, setStatus] = useState('Checking for an existing chat...')
  const [isBusy, setIsBusy] = useState(false)
  const [isEnded, setIsEnded] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let ignore = false

    async function restoreCurrentSession() {
      setIsBusy(true)

      try {
        const response = await fetch(`${API_BASE_URL}/api/sessions/current`, {
          credentials: 'include',
        })

        if (response.status === 404) {
          if (!ignore) {
            setHasSession(false)
            setStatus('Start a new chat')
          }
          return
        }

        if (!response.ok) {
          throw new Error(await readError(response))
        }

        const payload = (await response.json()) as GetSessionResponse

        if (!ignore) {
          setHasSession(true)
          setMessages(payload.messages.map(toChatMessage))
          setSummary(payload.summary)
          setIsEnded(Boolean(payload.summary))
          setStatus(payload.summary ? 'Session ended' : 'Ready')
        }
      } catch (error) {
        if (!ignore) {
          setStatus(error instanceof Error ? error.message : 'Could not restore chat')
        }
      } finally {
        if (!ignore) {
          setIsBusy(false)
        }
      }
    }

    restoreCurrentSession()

    return () => {
      ignore = true
    }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function startSession() {
    setIsBusy(true)
    setStatus('Starting session...')
    setSummary(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions`, {
        method: 'POST',
        credentials: 'include',
      })

      if (!response.ok) {
        throw new Error(await readError(response))
      }

      const payload = (await response.json()) as CreateSessionResponse

      setHasSession(true)
      setIsEnded(false)
      setMessages([
        {
          id: messageId(),
          role: 'assistant',
          text: payload.opening_message,
        },
      ])
      setStatus('Ready')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not start session')
    } finally {
      setIsBusy(false)
    }
  }

  async function endSession() {
    if (!hasSession || isBusy) {
      return
    }

    setIsBusy(true)
    setStatus('Ending session...')

    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions/current/end`, {
        method: 'POST',
        credentials: 'include',
      })

      if (!response.ok) {
        throw new Error(await readError(response))
      }

      const payload = (await response.json()) as EndSessionResponse
      setSummary(payload.summary)
      setHasSession(false)
      setIsEnded(true)
      setStatus('Session ended')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not end session')
    } finally {
      setIsBusy(false)
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const text = input.trim()
    if (!text || !hasSession || isBusy || isEnded) {
      return
    }

    setInput('')
    setIsBusy(true)
    setStatus('Thinking...')

    setMessages((current) => [
      ...current,
      {
        id: messageId(),
        role: 'user',
        text,
      },
    ])

    try {
      const response = await fetch(`${API_BASE_URL}/api/messages`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) {
        throw new Error(await readError(response))
      }

      const payload = (await response.json()) as SendMessageResponse

      setMessages((current) => [
        ...current,
        {
          id: messageId(),
          role: 'assistant',
          text: payload.text,
        },
      ])
      setStatus('Ready')
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: messageId(),
          role: 'assistant',
          text: error instanceof Error ? error.message : 'Something went wrong',
        },
      ])
      setStatus('Error')
    } finally {
      setIsBusy(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="chat-panel" aria-label="Simple chat">
        <header className="chat-header">
          <div>
            <h1>Simple Chat</h1>
            <p>{status}</p>
          </div>
          <div className="session-chip">{hasSession ? 'Connected' : 'No active chat'}</div>
        </header>

        <section className="session-controls" aria-label="Session controls">
          <button disabled={isBusy} onClick={startSession} type="button">
            Start New Chat
          </button>
          <button
            className="secondary"
            disabled={isBusy || !hasSession}
            onClick={endSession}
            type="button"
          >
            End Chat
          </button>
        </section>

        <div className="message-list">
          {messages.length === 0 && (
            <div className="empty-state">
              Start a new chat. If you already had an active chat in this browser, it
              will restore automatically after refresh.
            </div>
          )}

          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="message-label">
                {message.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div className="message-bubble">{message.text}</div>
            </article>
          ))}

          {isBusy && hasSession && (
            <article className="message assistant">
              <div className="message-label">Assistant</div>
              <div className="message-bubble muted">...</div>
            </article>
          )}

          {summary && (
            <section className="summary-panel">
              <h2>Session Summary</h2>
              <p>{summary}</p>
            </section>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            aria-label="Message"
            disabled={!hasSession || isBusy || isEnded}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }}
            placeholder={isEnded ? 'This session has ended' : 'Type a message...'}
            rows={1}
            value={input}
          />
          <button disabled={!hasSession || !input.trim() || isBusy || isEnded} type="submit">
            Send
          </button>
        </form>
      </section>
    </main>
  )
}

export default App
