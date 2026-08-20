import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { PipecatClient, RTVIEvent } from '@pipecat-ai/client-js'
import type { BotLLMTextData, TranscriptData } from '@pipecat-ai/client-js'
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport'
import { WavMediaManager, WebSocketTransport } from '@pipecat-ai/websocket-transport'
import './App.css'
import './SimpleChat.css'
import ScenarioDialog from "./ScenarioDialog"

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
  model_weights: string
  task_config: {
    lora_adapter?: string | null
    role?: string
    instructions?: string
  }
  messages: StoredSessionMessage[]
}

type LoraAdapter = {
  name: string
  loaded: boolean | null
  valid: boolean
  error: string | null
}

type LoraAdapterListResponse = {
  adapters: LoraAdapter[]
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

type Scenario = {
  name: string
  role: string
  instructions: string
}

type RealtimeVoiceState = 'disconnected' | 'connecting' | 'listening' | 'thinking' | 'speaking'
type RealtimeTransport = 'webrtc' | 'websocket'
type RealtimeTransportPreference = 'auto' | RealtimeTransport

type TurnTakingConfig = {
  enabled: boolean
  enable_interruptions: boolean
  min_words_to_interrupt: number
  use_interim_transcripts_for_interruptions: boolean
  vad_confidence: number
  vad_start_secs: number
  vad_stop_secs: number
  vad_min_volume: number
  smart_turn_stop_secs: number
  smart_turn_pre_speech_ms: number
  smart_turn_max_duration_secs: number
  smart_turn_cpu_threads: number
  user_turn_stop_timeout_secs: number
  audio_input_sample_rate: 16000
  audio_output_sample_rate: number
  output_audio_chunk_ms: number
  asr_language: string
  asr_automatic_punctuation: boolean
  asr_boosted_words: string[]
  asr_boost_score: number
}

const FALLBACK_TURN_TAKING_CONFIG: TurnTakingConfig = {
  enabled: true,
  enable_interruptions: true,
  min_words_to_interrupt: 3,
  use_interim_transcripts_for_interruptions: true,
  vad_confidence: 0.7,
  vad_start_secs: 0.2,
  vad_stop_secs: 0.2,
  vad_min_volume: 0.6,
  smart_turn_stop_secs: 3,
  smart_turn_pre_speech_ms: 500,
  smart_turn_max_duration_secs: 8,
  smart_turn_cpu_threads: 1,
  user_turn_stop_timeout_secs: 5,
  audio_input_sample_rate: 16000,
  audio_output_sample_rate: 24000,
  output_audio_chunk_ms: 20,
  asr_language: 'en-US',
  asr_automatic_punctuation: true,
  asr_boosted_words: [],
  asr_boost_score: 4,
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
  ?? import.meta.env.BASE_URL.replace(/\/$/, '')
const HISTORY_URL = `${import.meta.env.BASE_URL}history`
const EVALUATION_URL = "/evaluation/"
const RETRYABLE_STATUS_CODES = new Set([408, 409, 425, 429, 500, 502, 503, 504])

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(message)), timeoutMs)
    promise.then(
      (value) => { window.clearTimeout(timer); resolve(value) },
      (error) => { window.clearTimeout(timer); reject(error) },
    )
  })
}

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
  const [loraAdapters, setLoraAdapters] = useState<LoraAdapter[]>([])
  const [selectedAdapter, setSelectedAdapter] = useState('')
  const [adapterNotice, setAdapterNotice] = useState('Loading model options...')
  const [defaultScenario, setDefaultScenario] = useState<Scenario | null>(null)
  const [scenarioMode, setScenarioMode] = useState<"default" | "custom">("default")
  const [customRole, setCustomRole] = useState("")
  const [customInstructions, setCustomInstructions] = useState("")
  const [activeScenario, setActiveScenario] = useState<Scenario | null>(null)
  const [isScenarioOpen, setIsScenarioOpen] = useState(false)
  const [realtimeVoiceState, setRealtimeVoiceState] = useState<RealtimeVoiceState>('disconnected')
  const [interimTranscript, setInterimTranscript] = useState('')
  const [turnTakingConfig, setTurnTakingConfig] = useState<TurnTakingConfig>(FALLBACK_TURN_TAKING_CONFIG)
  const [realtimeIceServers, setRealtimeIceServers] = useState<RTCIceServer[]>([])
  const [realtimeTransportPreference, setRealtimeTransportPreference] = useState<RealtimeTransportPreference>('auto')
  const [activeRealtimeTransport, setActiveRealtimeTransport] = useState<RealtimeTransport | null>(null)
  const [webrtcConnectTimeoutMs, setWebrtcConnectTimeoutMs] = useState(5000)
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
  const realtimeClientRef = useRef<PipecatClient | null>(null)
  const realtimeAudioRef = useRef<HTMLAudioElement | null>(null)
  const realtimeVoiceStateRef = useRef<RealtimeVoiceState>('disconnected')
  const realtimeConnectionAttemptRef = useRef(0)

  useEffect(() => {
    let ignore = false

    async function loadTurnTakingConfig() {
      try {
        const response = await fetchWithRetry(`${API_BASE_URL}/api/realtime/config`)
        if (!response.ok) throw new Error(await readError(response))
        const payload = (await response.json()) as {
          defaults: TurnTakingConfig
          transport: RealtimeTransportPreference
          webrtc_connect_timeout_ms: number
          public_ice_servers: RTCIceServer[]
        }
        if (!ignore) {
          setTurnTakingConfig(payload.defaults)
          setRealtimeTransportPreference(payload.transport ?? 'auto')
          setWebrtcConnectTimeoutMs(payload.webrtc_connect_timeout_ms ?? 5000)
          setRealtimeIceServers(payload.public_ice_servers ?? [])
        }
      } catch {
        // The in-app defaults mirror the server and keep text/upload fallback usable.
      }
    }

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
          setSelectedAdapter(payload.task_config.lora_adapter ?? '')
          setActiveScenario({
            name: "Session scenario",
            role: payload.task_config.role ?? "",
            instructions: payload.task_config.instructions ?? "",
          })
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

    async function loadLoraAdapters() {
      try {
        const response = await fetchWithRetry(`${API_BASE_URL}/api/loras`)

        if (!response.ok) {
          throw new Error(await readError(response))
        }

        const payload = (await response.json()) as LoraAdapterListResponse
        if (!ignore) {
          setLoraAdapters(payload.adapters)
          setAdapterNotice(
            payload.adapters.some((adapter) => !adapter.valid)
              ? 'Some adapters are unavailable'
              : 'Applies to the next chat',
          )
        }
      } catch (error) {
        if (!ignore) {
          setAdapterNotice(
            error instanceof Error ? `Adapters unavailable: ${error.message}` : 'Adapters unavailable',
          )
        }
      }
    }

    async function loadDefaultScenario() {
      try {
        const response = await fetchWithRetry(API_BASE_URL + "/api/scenario")
        if (!response.ok) throw new Error(await readError(response))
        const payload = (await response.json()) as { default: Scenario }
        if (!ignore) {
          setDefaultScenario(payload.default)
          setCustomRole(payload.default.role)
          setCustomInstructions(payload.default.instructions)
        }
      } catch (error) {
        if (!ignore) {
          setStatus(error instanceof Error ? error.message : "Could not load scenario")
        }
      }
    }

    restoreCurrentSession()
    loadTurnTakingConfig()
    loadLoraAdapters()
    loadDefaultScenario()

    return () => {
      ignore = true
      stopAvatarAudio()
      stopRecordingResources()
      const realtimeClient = realtimeClientRef.current
      realtimeConnectionAttemptRef.current += 1
      realtimeClientRef.current = null
      if (realtimeClient) {
        void realtimeClient.disconnect().catch(() => undefined)
      }
    }
  }, [])

  const hasActiveSession = hasSession && !isEnded
  const configuredScenario: Scenario | null = defaultScenario && scenarioMode === "custom"
    ? { name: "Custom scenario", role: customRole, instructions: customInstructions }
    : defaultScenario
  const displayedScenario = hasActiveSession && activeScenario
    ? activeScenario
    : configuredScenario

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
    await disconnectRealtimeVoice()

    const audioUrl = audioBase64ToUrl(audioBase64, audioMimeType)
    const audio = new Audio(audioUrl)
    avatarAudioRef.current = audio
    avatarAudioUrlRef.current = audioUrl

    audio.onended = () => {
      stopAvatarAudio()
    }

    await audio.play()
  }

  function updateRealtimeVoiceState(next: RealtimeVoiceState) {
    realtimeVoiceStateRef.current = next
    setRealtimeVoiceState(next)
  }

  async function applyRealtimeMicConstraints(client: PipecatClient) {
    const track = client.tracks().local.audio
    if (!track) return

    try {
      await track.applyConstraints({
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      })
    } catch {
      // Some browsers expose WebRTC audio processing but reject explicit constraints.
    }
  }

  async function requestRealtimeMicPermission() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser does not support microphone capture.')
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    })
    stream.getTracks().forEach((track) => track.stop())
  }

  async function safelyDisconnectClient(client: PipecatClient) {
    try {
      await withTimeout(client.disconnect(), 1500, 'Disconnect timed out')
    } catch {
      // The failed transport may already be closed or still finishing ICE cleanup.
    }
  }

  async function disconnectRealtimeVoice() {
    realtimeConnectionAttemptRef.current += 1
    const client = realtimeClientRef.current
    realtimeClientRef.current = null
    setInterimTranscript('')
    setActiveRealtimeTransport(null)

    if (realtimeAudioRef.current) {
      realtimeAudioRef.current.srcObject = null
    }

    if (client) {
      await safelyDisconnectClient(client)
    }

    updateRealtimeVoiceState('disconnected')
  }

  async function startRealtimeVoice(speakOpeningMessage = false) {
    if (realtimeClientRef.current || !turnTakingConfig.enabled) return

    stopAvatarAudio()
    await disconnectRealtimeVoice()
    const attempt = realtimeConnectionAttemptRef.current + 1
    realtimeConnectionAttemptRef.current = attempt
    updateRealtimeVoiceState('connecting')
    setStatus('Waiting for microphone permission...')

    const isCurrent = (client?: PipecatClient) => (
      realtimeConnectionAttemptRef.current === attempt
      && (!client || realtimeClientRef.current === client)
    )

    const connectTransport = async (kind: RealtimeTransport) => {
      const transport = kind === 'webrtc'
        ? new SmallWebRTCTransport({ iceServers: realtimeIceServers })
        : new WebSocketTransport({
          mediaManager: new WavMediaManager(
            undefined,
            turnTakingConfig.audio_input_sample_rate,
          ),
        })
      const client = new PipecatClient({ transport, enableMic: true, enableCam: false })
      realtimeClientRef.current = client

      client.on(RTVIEvent.TrackStarted, (track, participant) => {
        if (!isCurrent(client) || track.kind !== 'audio') return
        if (participant?.local) {
          void applyRealtimeMicConstraints(client)
          return
        }
        if (realtimeAudioRef.current) {
          realtimeAudioRef.current.srcObject = new MediaStream([track])
          void realtimeAudioRef.current.play().catch(() => undefined)
        }
      })
      client.on(RTVIEvent.UserTranscript, (data: TranscriptData) => {
        if (isCurrent(client)) setInterimTranscript(data.final ? '' : data.text)
      })
      client.on(RTVIEvent.UserLlmText, (data) => {
        if (!isCurrent(client)) return
        const text = data.text.trim()
        if (!text) return
        setInterimTranscript('')
        setMessages((current) => [...current, { id: messageId(), role: 'user', text }])
      })
      client.on(RTVIEvent.BotLlmText, (data: BotLLMTextData) => {
        if (!isCurrent(client)) return
        const text = data.text.trim()
        if (text) setMessages((current) => [...current, { id: messageId(), role: 'assistant', text }])
      })
      client.on(RTVIEvent.UserStartedSpeaking, () => {
        if (!isCurrent(client)) return
        const wasSpeaking = realtimeVoiceStateRef.current === 'speaking'
        updateRealtimeVoiceState('listening')
        setStatus(wasSpeaking ? 'Interrupted — listening...' : 'Listening...')
      })
      client.on(RTVIEvent.UserStoppedSpeaking, () => {
        if (!isCurrent(client)) return
        updateRealtimeVoiceState('thinking')
        setStatus('Thinking...')
      })
      client.on(RTVIEvent.BotStartedSpeaking, () => {
        if (!isCurrent(client)) return
        updateRealtimeVoiceState('speaking')
        setStatus('Speaking — you can interrupt')
      })
      client.on(RTVIEvent.BotStoppedSpeaking, () => {
        if (isCurrent(client) && realtimeVoiceStateRef.current === 'speaking') {
          updateRealtimeVoiceState('listening')
          setStatus('Listening continuously')
        }
      })
      client.on(RTVIEvent.Disconnected, () => {
        if (!isCurrent(client)) return
        realtimeClientRef.current = null
        setActiveRealtimeTransport(null)
        updateRealtimeVoiceState('disconnected')
        setInterimTranscript('')
        setStatus('Realtime voice disconnected')
      })
      client.on(RTVIEvent.Error, (message) => {
        if (!isCurrent(client)) return
        const data = message.data as { message?: string } | undefined
        setStatus(data?.message ?? 'Realtime voice error')
      })

      await withTimeout(
        client.initDevices(),
        8000,
        `${kind === 'webrtc' ? 'WebRTC' : 'WebSocket'} audio device initialization timed out`,
      )
      if (!isCurrent(client)) throw new Error('Connection cancelled')
      await applyRealtimeMicConstraints(client)

      if (kind === 'webrtc') {
        await withTimeout(client.connect({
          webrtcRequestParams: {
            endpoint: `${API_BASE_URL}/api/realtime/offer`,
            requestData: { turn_taking: turnTakingConfig, speak_opening_message: speakOpeningMessage },
          },
        }), webrtcConnectTimeoutMs, 'WebRTC media connection timed out')
      } else {
        const url = new URL(`${API_BASE_URL}/api/realtime/ws`, window.location.origin)
        url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
        url.searchParams.set('turn_taking', JSON.stringify(turnTakingConfig))
        url.searchParams.set('speak_opening_message', String(speakOpeningMessage))
        await withTimeout(
          client.connect({ wsUrl: url.toString() }),
          8000,
          'WebSocket voice connection timed out',
        )
      }
      if (!isCurrent(client)) throw new Error('Connection cancelled')
      setActiveRealtimeTransport(kind)
      updateRealtimeVoiceState('listening')
      setStatus(`Listening continuously (${kind === 'webrtc' ? 'WebRTC' : 'WebSocket'})`)
    }

    try {
      await requestRealtimeMicPermission()
      if (realtimeConnectionAttemptRef.current !== attempt) return

      const transports: RealtimeTransport[] = realtimeTransportPreference === 'auto'
        ? ['webrtc', 'websocket']
        : [realtimeTransportPreference]
      let lastError: unknown
      for (const kind of transports) {
        if (realtimeConnectionAttemptRef.current !== attempt) return
        setStatus(kind === 'webrtc' ? 'Connecting WebRTC voice...' : 'Connecting tunnel-compatible voice...')
        try {
          await connectTransport(kind)
          return
        } catch (error) {
          lastError = error
          const failedClient = realtimeClientRef.current
          realtimeClientRef.current = null
          if (failedClient) await safelyDisconnectClient(failedClient)
          if (kind === 'webrtc' && realtimeTransportPreference === 'auto') {
            setStatus('WebRTC unavailable; switching to WebSocket...')
          }
        }
      }
      throw lastError ?? new Error('Could not connect realtime voice')
    } catch (error) {
      if (realtimeConnectionAttemptRef.current === attempt) {
        const failedClient = realtimeClientRef.current
        realtimeClientRef.current = null
        if (failedClient) await safelyDisconnectClient(failedClient)
        setActiveRealtimeTransport(null)
        updateRealtimeVoiceState('disconnected')
        setStatus(error instanceof Error ? error.message : 'Could not connect realtime voice')
      }
    }
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
    await disconnectRealtimeVoice()
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
          synthesize_audio: mode !== 'voice',
          lora_adapter: selectedAdapter || null,
          scenario: scenarioMode === "custom"
            ? { role: customRole, instructions: customInstructions }
            : null,
        }),
      })

      if (!response.ok) {
        throw new Error(await readError(response))
      }

      const payload = (await response.json()) as CreateSessionResponse

      setHasSession(true)
      setIsEnded(false)
      setActiveScenario(configuredScenario)
      setMessages([
        {
          id: messageId(),
          role: 'assistant',
          text: payload.opening_message,
        },
      ])
      setStatus('Ready')
      if (mode === 'voice') {
        void startRealtimeVoice(true)
      } else {
        await playAvatarSpeech(payload.audio_base64, payload.audio_mime_type)
      }
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
    await disconnectRealtimeVoice()
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
    await disconnectRealtimeVoice()
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
    await disconnectRealtimeVoice()
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
      void disconnectRealtimeVoice()
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
      <section className="chat-panel" aria-label="Simple Chat">
        <header className="chat-header">
          <div>
            <div className="title-line">
              <h1>Simple Chat</h1>
            </div>
            <p>{status}</p>
          </div>
          <div className="header-actions">
            <a className="nav-button secondary" href={HISTORY_URL}>Session history</a>
            <a className="nav-button secondary" href={EVALUATION_URL}>Evaluation</a>
            <button
              className="nav-button secondary scenario-trigger"
              disabled={!displayedScenario}
              onClick={() => setIsScenarioOpen(true)}
              type="button"
            >
              Scenario
            </button>
            <div className="session-chip">{hasActiveSession ? 'Connected' : 'No active chat'}</div>
          </div>
        </header>

        <section className="session-controls" aria-label="Session controls">
          <button
            disabled={isBusy || isRecording || hasActiveSession || (scenarioMode === "custom" && (!customRole.trim() || !customInstructions.trim()))}
            onClick={startSession}
            type="button"
          >
            Start New Chat
          </button>
          <button
            className="secondary"
            disabled={isBusy || isRecording || !hasActiveSession}
            onClick={endSession}
            type="button"
          >
            End Chat
          </button>
          <label className="adapter-selector">
            <span>Model weights</span>
            <select
              aria-describedby="adapter-notice"
              disabled={hasActiveSession || isBusy || isRecording}
              onChange={(event) => setSelectedAdapter(event.target.value)}
              value={selectedAdapter}
            >
              <option value="">Base model (no adapter)</option>
              {selectedAdapter && !loraAdapters.some((adapter) => adapter.name === selectedAdapter) && (
                <option value={selectedAdapter}>{selectedAdapter}</option>
              )}
              {loraAdapters.map((adapter) => (
                <option disabled={!adapter.valid} key={adapter.name} value={adapter.name}>
                  {adapter.name}{adapter.loaded ? ' (loaded)' : ''}{adapter.valid ? '' : ' (unavailable)'}
                </option>
              ))}
            </select>
            <small id="adapter-notice">
              {hasActiveSession ? 'End the active chat to change weights' : adapterNotice}
            </small>
          </label>
          <div className="mode-toggle" role="group" aria-label="Input mode">
            <button
              className={mode === 'text' ? 'active' : ''}
              disabled={isRecording}
              onClick={() => { void disconnectRealtimeVoice(); setMode('text') }}
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
          <section className="voice-composer realtime" aria-label="Realtime voice controls">
            <audio autoPlay className="realtime-audio" ref={realtimeAudioRef} />
            <div className={`voice-state ${realtimeVoiceState}`}>
              <span aria-hidden="true" className="voice-state-dot" />
              <div>
                <strong>{realtimeVoiceState}</strong>
                <small>
                  {interimTranscript || (activeRealtimeTransport
                    ? `${activeRealtimeTransport === 'webrtc' ? 'WebRTC' : 'WebSocket'} · ${realtimeVoiceState === 'speaking'
                      ? `say ${turnTakingConfig.min_words_to_interrupt}+ words to interrupt`
                      : 'mic stays open; Smart Turn handles natural pauses'}`
                    : (realtimeVoiceState === 'speaking'
                    ? `Say ${turnTakingConfig.min_words_to_interrupt}+ words to interrupt`
                    : 'Mic stays open; Smart Turn handles natural pauses'))}
                </small>
              </div>
            </div>
            <button
              className={realtimeVoiceState === 'disconnected' ? '' : 'recording'}
              disabled={!hasSession || isBusy || isEnded || !turnTakingConfig.enabled}
              onClick={() => {
                if (realtimeVoiceState === 'disconnected') void startRealtimeVoice()
                else void disconnectRealtimeVoice()
              }}
              type="button"
            >
              {realtimeVoiceState === 'disconnected' ? 'Connect Realtime Voice' : 'Disconnect Voice'}
            </button>
            <details className="turn-settings">
              <summary>Turn-taking settings</summary>
              <div className="turn-settings-grid">
                <label>
                  <span>Audio transport</span>
                  <select
                    disabled={realtimeVoiceState !== 'disconnected'}
                    onChange={(event) => setRealtimeTransportPreference(event.target.value as RealtimeTransportPreference)}
                    value={realtimeTransportPreference}
                  >
                    <option value="auto">Auto (recommended)</option>
                    <option value="webrtc">WebRTC only</option>
                    <option value="websocket">WebSocket only</option>
                  </select>
                </label>
                <label>
                  <span>Words to interrupt</span>
                  <input
                    disabled={realtimeVoiceState !== 'disconnected'}
                    max={20}
                    min={1}
                    onChange={(event) => setTurnTakingConfig((current) => ({
                      ...current,
                      min_words_to_interrupt: Number(event.target.value),
                    }))}
                    type="number"
                    value={turnTakingConfig.min_words_to_interrupt}
                  />
                </label>
                <label>
                  <span>VAD confidence</span>
                  <input
                    disabled={realtimeVoiceState !== 'disconnected'}
                    max={0.99}
                    min={0.1}
                    onChange={(event) => setTurnTakingConfig((current) => ({
                      ...current,
                      vad_confidence: Number(event.target.value),
                    }))}
                    step={0.05}
                    type="number"
                    value={turnTakingConfig.vad_confidence}
                  />
                </label>
                <label>
                  <span>Smart Turn max silence</span>
                  <input
                    disabled={realtimeVoiceState !== 'disconnected'}
                    max={10}
                    min={0.5}
                    onChange={(event) => setTurnTakingConfig((current) => ({
                      ...current,
                      smart_turn_stop_secs: Number(event.target.value),
                    }))}
                    step={0.5}
                    type="number"
                    value={turnTakingConfig.smart_turn_stop_secs}
                  />
                </label>
              </div>
              <p>
                Browser echo cancellation, noise suppression and automatic gain control are enabled.
                Changes apply on the next connection.
              </p>
            </details>
            <details className="upload-fallback">
              <summary>Upload recording fallback</summary>
              <canvas className="waveform" height={48} ref={waveformCanvasRef} width={360} />
              <button
                className={isRecording ? 'recording' : ''}
                disabled={!hasSession || isBusy || isEnded || realtimeVoiceState !== 'disconnected'}
                onClick={isRecording ? stopRecording : startRecording}
                type="button"
              >
                {isRecording ? 'Stop Recording' : 'Record One Message'}
              </button>
            </details>
          </section>
        )}
      </section>
      <ScenarioDialog
        activeScenario={activeScenario}
        customInstructions={customInstructions}
        customRole={customRole}
        defaultScenario={defaultScenario}
        isActiveSession={hasActiveSession}
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
