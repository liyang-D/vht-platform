import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type ChatMode = 'text' | 'voice'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  text: string
}

type CreateSessionResponse = {
  opening_message: string
  audio_base64: string | null
  audio_mime_type: string | null
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
  audio_base64: string | null
  audio_mime_type: string | null
}

type TranscribeAudioResponse = {
  transcript: string
}

type EndSessionResponse = {
  summary: string
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''
const RETRYABLE_STATUS_CODES = new Set([408, 409, 425, 429, 500, 502, 503, 504])

function messageId() {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID()
  }

  return `msg-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

function toChatMessage(message: StoredSessionMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role === 'user' ? 'user' : 'assistant',
    text: message.text,
  }
}

function audioBase64ToUrl(base64: string, mimeType: string) {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return URL.createObjectURL(new Blob([bytes], { type: mimeType }))
}

function writeAscii(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index))
  }
}

function resampleMono(buffer: AudioBuffer, targetSampleRate: number) {
  const input = buffer.getChannelData(0)
  const ratio = buffer.sampleRate / targetSampleRate
  const outputLength = Math.max(1, Math.round(input.length / ratio))
  const output = new Float32Array(outputLength)

  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio
    const leftIndex = Math.floor(position)
    const rightIndex = Math.min(leftIndex + 1, input.length - 1)
    const fraction = position - leftIndex
    output[index] = input[leftIndex] * (1 - fraction) + input[rightIndex] * fraction
  }

  return output
}

function encodeWav(samples: Float32Array, sampleRate: number) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)

  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeAscii(view, 8, 'WAVE')
  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeAscii(view, 36, 'data')
  view.setUint32(40, samples.length * 2, true)

  let offset = 44
  for (const sample of samples) {
    const clipped = Math.max(-1, Math.min(1, sample))
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true)
    offset += 2
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

async function convertToWav(audioBlob: Blob) {
  const audioContext = new AudioContext()

  try {
    const decoded = await audioContext.decodeAudioData(await audioBlob.arrayBuffer())
    const sampleRate = 16000
    return encodeWav(resampleMono(decoded, sampleRate), sampleRate)
  } finally {
    void audioContext.close()
  }
}

async function readError(response: Response) {
  try {
    const payload = await response.json()
    const detail = payload.detail ?? payload.error ?? payload.message

    if (typeof detail === 'string') {
      return detail
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (typeof item === 'string') {
            return item
          }

          if (item && typeof item === 'object' && 'msg' in item) {
            return String(item.msg)
          }

          return JSON.stringify(item)
        })
        .join('; ')
    }

    if (detail && typeof detail === 'object') {
      return JSON.stringify(detail)
    }

    return 'Request failed'
  } catch {
    return 'Request failed'
  }
}

function sleep(milliseconds: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds)
  })
}

async function fetchWithRetry(input: RequestInfo | URL, init?: RequestInit) {
  const maxAttempts = 3
  let lastError: unknown = null

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      const response = await fetch(input, init)

      if (!RETRYABLE_STATUS_CODES.has(response.status) || attempt === maxAttempts - 1) {
        return response
      }
    } catch (error) {
      lastError = error

      if (attempt === maxAttempts - 1) {
        throw error
      }
    }

    await sleep(400 * 2 ** attempt)
  }

  throw lastError instanceof Error ? lastError : new Error('Request failed')
}

function App() {
  const [mode, setMode] = useState<ChatMode>('text')
  const [hasSession, setHasSession] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [summary, setSummary] = useState<string | null>(null)
  const [status, setStatus] = useState('Checking for an existing chat...')
  const [isBusy, setIsBusy] = useState(false)
  const [isEnded, setIsEnded] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<BlobPart[]>([])
  const audioStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const waveformCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const avatarAudioRef = useRef<HTMLAudioElement | null>(null)
  const avatarAudioUrlRef = useRef<string | null>(null)

  useEffect(() => {
    let ignore = false

    async function restoreCurrentSession() {
      setIsBusy(true)

      try {
        const response = await fetchWithRetry(`${API_BASE_URL}/api/sessions/current`, {
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
      stopAvatarAudio()
      stopRecordingResources()
    }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function stopAvatarAudio() {
    if (avatarAudioRef.current) {
      avatarAudioRef.current.pause()
      avatarAudioRef.current.currentTime = 0
      avatarAudioRef.current = null
    }

    if (avatarAudioUrlRef.current) {
      URL.revokeObjectURL(avatarAudioUrlRef.current)
      avatarAudioUrlRef.current = null
    }
  }

  async function playAvatarSpeech(audioBase64: string | null, audioMimeType: string | null) {
    if (!audioBase64 || !audioMimeType) {
      return
    }

    stopAvatarAudio()

    const audioUrl = audioBase64ToUrl(audioBase64, audioMimeType)
    const audio = new Audio(audioUrl)
    avatarAudioRef.current = audio
    avatarAudioUrlRef.current = audioUrl

    audio.onended = () => {
      stopAvatarAudio()
    }

    await audio.play()
  }

  function stopRecordingResources() {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    if (audioContextRef.current) {
      void audioContextRef.current.close()
      audioContextRef.current = null
    }

    audioStreamRef.current?.getTracks().forEach((track) => track.stop())
    audioStreamRef.current = null
    analyserRef.current = null
  }

  function drawWaveform() {
    const canvas = waveformCanvasRef.current
    const analyser = analyserRef.current

    if (!canvas || !analyser) {
      return
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    const data = new Uint8Array(analyser.frequencyBinCount)
    analyser.getByteTimeDomainData(data)

    context.clearRect(0, 0, canvas.width, canvas.height)
    context.lineWidth = 2
    context.strokeStyle = '#2563eb'
    context.beginPath()

    const sliceWidth = canvas.width / data.length

    for (let index = 0; index < data.length; index += 1) {
      const x = index * sliceWidth
      const y = (data[index] / 255) * canvas.height

      if (index === 0) {
        context.moveTo(x, y)
      } else {
        context.lineTo(x, y)
      }
    }

    context.stroke()
    animationFrameRef.current = requestAnimationFrame(drawWaveform)
  }

  async function startSession() {
    stopAvatarAudio()
    setIsBusy(true)
    setStatus('Starting session...')
    setSummary(null)

    try {
      const response = await fetchWithRetry(`${API_BASE_URL}/api/sessions`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          response_modality: mode,
        }),
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
      await playAvatarSpeech(payload.audio_base64, payload.audio_mime_type)
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

    stopAvatarAudio()
    setIsBusy(true)
    setStatus('Ending session...')

    try {
      const response = await fetchWithRetry(`${API_BASE_URL}/api/sessions/current/end`, {
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

    stopAvatarAudio()
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
      const response = await fetchWithRetry(`${API_BASE_URL}/api/messages`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          response_modality: 'text',
        }),
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

  async function startRecording() {
    if (!hasSession || isBusy || isEnded || isRecording) {
      return
    }

    stopAvatarAudio()
    setStatus('Recording...')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const audioContext = new AudioContext()
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()

      analyser.fftSize = 2048
      source.connect(analyser)

      audioStreamRef.current = stream
      audioContextRef.current = audioContext
      analyserRef.current = analyser
      audioChunksRef.current = []

      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      })

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        })
        audioChunksRef.current = []
        stopRecordingResources()
        void sendAudioBlob(audioBlob)
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
      drawWaveform()
    } catch (error) {
      stopRecordingResources()
      setIsRecording(false)
      setStatus(error instanceof Error ? error.message : 'Could not access microphone')
    }
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorderRef.current) {
      return
    }

    setIsRecording(false)
    setStatus('Processing voice...')
    mediaRecorderRef.current.stop()
    mediaRecorderRef.current = null
  }

  async function sendAudioBlob(audioBlob: Blob) {
    if (!hasSession || audioBlob.size === 0) {
      setStatus('Ready')
      return
    }

    setIsBusy(true)

    try {
      const wavBlob = await convertToWav(audioBlob)
      const formData = new FormData()
      formData.append('audio', wavBlob, 'voice-message.wav')

      const transcriptionResponse = await fetchWithRetry(`${API_BASE_URL}/api/audio-transcriptions`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      })

      if (!transcriptionResponse.ok) {
        throw new Error(await readError(transcriptionResponse))
      }

      const transcription = (await transcriptionResponse.json()) as TranscribeAudioResponse

      setMessages((current) => [
        ...current,
        {
          id: messageId(),
          role: 'user',
          text: transcription.transcript,
        },
      ])
      setStatus('Thinking...')

      const response = await fetchWithRetry(`${API_BASE_URL}/api/messages`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: transcription.transcript,
          response_modality: 'voice',
        }),
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
      await playAvatarSpeech(payload.audio_base64, payload.audio_mime_type)
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
          <button disabled={isBusy || isRecording} onClick={startSession} type="button">
            Start New Chat
          </button>
          <button
            className="secondary"
            disabled={isBusy || isRecording || !hasSession}
            onClick={endSession}
            type="button"
          >
            End Chat
          </button>
          <div className="mode-toggle" role="group" aria-label="Input mode">
            <button
              className={mode === 'text' ? 'active' : ''}
              disabled={isRecording}
              onClick={() => setMode('text')}
              type="button"
            >
              Text
            </button>
            <button
              className={mode === 'voice' ? 'active' : ''}
              disabled={isRecording}
              onClick={() => setMode('voice')}
              type="button"
            >
              Voice
            </button>
          </div>
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

        {mode === 'text' ? (
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              aria-label="Message"
              disabled={!hasSession || isBusy || isEnded || isRecording}
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
        ) : (
          <section className="voice-composer" aria-label="Voice message controls">
            <canvas
              className="waveform"
              height={48}
              ref={waveformCanvasRef}
              width={360}
            />
            <button
              className={isRecording ? 'recording' : ''}
              disabled={!hasSession || isBusy || isEnded}
              onClick={isRecording ? stopRecording : startRecording}
              type="button"
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>
          </section>
        )}
      </section>
    </main>
  )
}

export default App
