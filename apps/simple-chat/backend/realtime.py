"""Pipecat-based full-duplex voice transport for Simple Chat.

The realtime lane deliberately stays thin: Pipecat owns WebRTC media, VAD,
turn detection and interruption semantics, while VHT's existing orchestrator,
Riva ASR and Chatterbox TTS remain the inference backends.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal

import aiohttp
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

import orchestrator_client
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.settings import LLMSettings, TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.text.base_text_aggregator import AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.workers.runner import WorkerRunner


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class TurnTakingConfig(BaseModel):
    """Per-connection turn-taking config, with server defaults and safe bounds."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    enable_interruptions: bool = True
    min_words_to_interrupt: int = Field(default=3, ge=1, le=20)
    use_interim_transcripts_for_interruptions: bool = True

    vad_confidence: float = Field(default=0.7, ge=0.1, le=0.99)
    vad_start_secs: float = Field(default=0.3, ge=0.05, le=2.0)
    vad_stop_secs: float = Field(default=0.2, ge=0.05, le=2.0)
    vad_min_volume: float = Field(default=0.55, ge=0.0, le=1.0)

    smart_turn_stop_secs: float = Field(default=3.0, ge=0.5, le=10.0)
    smart_turn_pre_speech_ms: float = Field(default=500.0, ge=0.0, le=2000.0)
    smart_turn_max_duration_secs: float = Field(default=8.0, ge=2.0, le=8.0)
    smart_turn_cpu_threads: int = Field(default=1, ge=1, le=4)
    user_turn_stop_timeout_secs: float = Field(default=5.0, ge=1.0, le=20.0)

    audio_input_sample_rate: Literal[16000] = 16000
    audio_output_sample_rate: int = Field(default=24000, ge=16000, le=48000)
    output_audio_chunk_ms: int = Field(default=20, ge=10, le=100, multiple_of=10)

    asr_language: str = Field(default="en-US", min_length=2, max_length=32)
    asr_automatic_punctuation: bool = True
    asr_boosted_words: list[str] = Field(default_factory=list, max_length=100)
    asr_boost_score: float = Field(default=4.0, ge=0.0, le=100.0)

    @classmethod
    def server_defaults(cls) -> "TurnTakingConfig":
        return cls(
            enabled=_env_bool("TURN_TAKING_ENABLED", True),
            enable_interruptions=_env_bool("TURN_TAKING_ENABLE_INTERRUPTION", True),
            min_words_to_interrupt=int(os.getenv("TURN_TAKING_MIN_WORDS", "3")),
            use_interim_transcripts_for_interruptions=_env_bool(
                "TURN_TAKING_USE_INTERIM_TRANSCRIPTS", True
            ),
            vad_confidence=float(os.getenv("TURN_TAKING_VAD_CONFIDENCE", "0.7")),
            vad_start_secs=float(os.getenv("TURN_TAKING_VAD_START_SECS", "0.3")),
            vad_stop_secs=float(os.getenv("TURN_TAKING_VAD_STOP_SECS", "0.2")),
            vad_min_volume=float(os.getenv("TURN_TAKING_VAD_MIN_VOLUME", "0.55")),
            smart_turn_stop_secs=float(
                os.getenv("TURN_TAKING_SMART_TURN_STOP_SECS", "3.0")
            ),
            smart_turn_pre_speech_ms=float(
                os.getenv("TURN_TAKING_SMART_TURN_PRE_SPEECH_MS", "500")
            ),
            smart_turn_max_duration_secs=float(
                os.getenv("TURN_TAKING_SMART_TURN_MAX_DURATION_SECS", "8.0")
            ),
            smart_turn_cpu_threads=int(
                os.getenv("TURN_TAKING_SMART_TURN_CPU_THREADS", "1")
            ),
            user_turn_stop_timeout_secs=float(
                os.getenv("TURN_TAKING_USER_TURN_STOP_TIMEOUT_SECS", "5.0")
            ),
            audio_output_sample_rate=int(
                os.getenv("TURN_TAKING_AUDIO_OUTPUT_SAMPLE_RATE", "24000")
            ),
            output_audio_chunk_ms=int(
                os.getenv("TURN_TAKING_OUTPUT_AUDIO_CHUNK_MS", "20")
            ),
            asr_language=os.getenv("RIVA_ASR_LANGUAGE", "en-US"),
            asr_automatic_punctuation=_env_bool(
                "RIVA_ASR_ENABLE_AUTOMATIC_PUNCTUATION", True
            ),
        )

    @classmethod
    def resolve(cls, request_data: Any) -> "TurnTakingConfig":
        server_config = cls.server_defaults()
        defaults = server_config.model_dump()
        if not isinstance(request_data, dict):
            return server_config

        override = request_data.get("turn_taking", request_data)
        if not isinstance(override, dict):
            return server_config

        resolved = cls.model_validate({**defaults, **override})
        if not server_config.enabled:
            resolved.enabled = False
        return resolved


class OrchestratorLLMService(LLMService):
    """LLM frame adapter that keeps generation inside the VHT orchestrator."""

    def __init__(self, *, session_id: str):
        super().__init__(settings=LLMSettings(model="vht-orchestrator"))
        self._session_id = session_id

    @staticmethod
    def _latest_user_text(context: LLMContext) -> str:
        for message in reversed(context.messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    str(item.get("text", "")).strip()
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return " ".join(part for part in parts if part)
        return ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        try:
            user_text = self._latest_user_text(frame.context)
            if not user_text:
                raise RuntimeError("The completed user turn did not contain a transcript.")

            received_text = False
            async for event in orchestrator_client.stream_message(
                session_id=self._session_id,
                text=user_text,
                response_modality="voice",
            ):
                if event.get("type") != "text_delta":
                    continue
                text = str(event.get("text") or "")
                if not text:
                    continue
                if not received_text:
                    received_text = True
                    await self.stop_ttfb_metrics()
                await self._push_llm_text(text)

            if not received_text:
                raise RuntimeError("The orchestrator returned an empty response.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.push_error("Realtime orchestrator response failed", exception=exc)
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())


class ChatterboxHttpTTSService(TTSService):
    """Thin Pipecat adapter around the existing Chatterbox WAV endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        voice: str,
        aiohttp_session: aiohttp.ClientSession,
    ):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=float(
                os.getenv("SIMPLE_CHAT_REALTIME_TTS_STOP_FRAME_TIMEOUT_SECONDS", "30")
            ),
            settings=TTSSettings(model="chatterbox-turbo", voice=voice),
        )
        self._url = f"{base_url.rstrip('/')}/v1/audio/synthesize"
        self._session = aiohttp_session
        self._voice = voice

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            async with self._session.post(
                self._url,
                json={"text": text, "voice": self._voice},
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    yield ErrorFrame(
                        error=f"Chatterbox TTS failed with status {response.status}: {detail}"
                    )
                    return

                async def chunks() -> AsyncIterator[bytes]:
                    async for chunk in response.content.iter_chunked(self.chunk_size):
                        if chunk:
                            yield chunk

                first_frame = True
                async for audio_frame in self._stream_audio_frames_from_iterator(
                    chunks(),
                    strip_wav_header=True,
                    context_id=context_id,
                ):
                    if first_frame:
                        await self.stop_ttfb_metrics()
                        first_frame = False
                    yield audio_frame
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield ErrorFrame(error=f"Chatterbox TTS request failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()


def _language(value: str) -> Language:
    try:
        return Language(value)
    except ValueError:
        logger.warning("Unsupported ASR language {}; falling back to en-US", value)
        return Language.EN_US


def _ice_servers() -> list[IceServer] | list[str] | None:
    raw = os.getenv("TURN_TAKING_ICE_SERVERS_JSON", "").strip()
    if not raw:
        return None

    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("TURN_TAKING_ICE_SERVERS_JSON must be a JSON list.")

    servers: list[IceServer] = []
    for item in data:
        if isinstance(item, str):
            servers.append(IceServer(urls=item))
        elif isinstance(item, dict):
            servers.append(
                IceServer(
                    urls=item.get("urls"),
                    username=item.get("username"),
                    credential=item.get("credential"),
                )
            )
        else:
            raise ValueError("ICE server entries must be strings or objects.")
    return servers


def public_browser_ice_servers() -> list[dict[str, str]]:
    raw = os.getenv("TURN_TAKING_ICE_SERVERS_JSON", "").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("TURN_TAKING_ICE_SERVERS_JSON must be a JSON list.")
    return [{"urls": item} for item in data if isinstance(item, str)]


_webrtc_handler = SmallWebRTCRequestHandler(ice_servers=_ice_servers())
_active_tasks: set[asyncio.Task[None]] = set()
_session_tasks: dict[str, asyncio.Task[None]] = {}


async def _natural_tts_frames(text: str) -> list[TTSSpeakFrame]:
    """Split text only at Pipecat-confirmed sentence boundaries."""
    aggregator = SimpleTextAggregator(aggregation_type=AggregationType.SENTENCE)
    sentences: list[str] = []

    async for sentence in aggregator.aggregate(text):
        if sentence.text.strip():
            sentences.append(sentence.text.strip())

    remainder = await aggregator.flush()
    if remainder and remainder.text.strip():
        sentences.append(remainder.text.strip())

    return [
        TTSSpeakFrame(sentence, append_to_context=False)
        for sentence in sentences
    ]


async def _run_voice_pipeline(
    transport: BaseTransport,
    *,
    session_id: str,
    opening_message: str,
    config: TurnTakingConfig,
) -> None:
    timeout = aiohttp.ClientTimeout(
        total=float(os.getenv("SIMPLE_CHAT_REALTIME_TTS_TIMEOUT_SECONDS", "300"))
    )
    async with aiohttp.ClientSession(timeout=timeout) as http_session:
        vad = SileroVADAnalyzer(
            sample_rate=config.audio_input_sample_rate,
            params=VADParams(
                confidence=config.vad_confidence,
                start_secs=config.vad_start_secs,
                stop_secs=config.vad_stop_secs,
                min_volume=config.vad_min_volume,
            ),
        )
        smart_turn = LocalSmartTurnAnalyzerV3(
            sample_rate=config.audio_input_sample_rate,
            cpu_count=config.smart_turn_cpu_threads,
            params=SmartTurnParams(
                stop_secs=config.smart_turn_stop_secs,
                pre_speech_ms=config.smart_turn_pre_speech_ms,
                max_duration_secs=config.smart_turn_max_duration_secs,
            ),
        )

        stt = NvidiaSTTService(
            server=os.getenv("RIVA_ASR_GRPC_SERVER", "asr:50051"),
            use_ssl=False,
            model_function_map={
                "function_id": "",
                "model_name": os.getenv("ASR_MODEL", "parakeet-1-1b-ctc-en-us"),
            },
            sample_rate=config.audio_input_sample_rate,
            settings=NvidiaSTTService.Settings(
                language=_language(config.asr_language),
                automatic_punctuation=config.asr_automatic_punctuation,
                interim_results=True,
                boosted_lm_words=config.asr_boosted_words or None,
                boosted_lm_score=config.asr_boost_score,
            ),
        )
        llm = OrchestratorLLMService(session_id=session_id)
        tts = ChatterboxHttpTTSService(
            base_url=os.getenv("CHATTERBOX_TTS_HTTP_URL", "http://tts:8000"),
            voice=os.getenv("CHATTERBOX_TTS_VOICE", "default"),
            aiohttp_session=http_session,
        )

        context = LLMContext()
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=vad,
                user_turn_strategies=UserTurnStrategies(
                    start=[
                        MinWordsUserTurnStartStrategy(
                            min_words=config.min_words_to_interrupt,
                            use_interim=config.use_interim_transcripts_for_interruptions,
                            enable_interruptions=config.enable_interruptions,
                        )
                    ],
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)
                    ],
                ),
                user_turn_stop_timeout=config.user_turn_stop_timeout_secs,
            ),
        )

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_aggregator,
                llm,
                tts,
                transport.output(),
                assistant_aggregator,
            ]
        )
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=config.audio_input_sample_rate,
                audio_out_sample_rate=config.audio_output_sample_rate,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=int(os.getenv("TURN_TAKING_IDLE_TIMEOUT_SECS", "900")),
        )
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client):
            logger.info("Realtime voice client connected for session {}", session_id)
            if opening_message:
                await worker.queue_frames(await _natural_tts_frames(opening_message))

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client):
            logger.info("Realtime voice client disconnected for session {}", session_id)
            await runner.cancel()

        await runner.add_workers(worker)
        await runner.run()


def _opening_message(session: dict[str, Any]) -> str:
    for message in reversed(session.get("messages") or []):
        if message.get("role") in {"avatar", "assistant"}:
            return str(message.get("text") or "")
    return ""


def _track_task(task: asyncio.Task[None], session_id: str) -> None:
    _active_tasks.add(task)
    _session_tasks[session_id] = task

    def done(completed: asyncio.Task[None]) -> None:
        _active_tasks.discard(completed)
        if _session_tasks.get(session_id) is completed:
            _session_tasks.pop(session_id, None)
        if completed.cancelled():
            return
        error = completed.exception()
        if error:
            logger.error("Realtime voice pipeline failed: {}", error)

    task.add_done_callback(done)


async def _cancel_session_pipeline(session_id: str) -> None:
    previous_task = _session_tasks.get(session_id)
    if previous_task and previous_task is not asyncio.current_task() and not previous_task.done():
        previous_task.cancel()
        await asyncio.gather(previous_task, return_exceptions=True)


async def handle_offer(payload: dict[str, Any], session_id: str) -> dict[str, str] | None:
    session = await orchestrator_client.get_session(session_id)
    request = SmallWebRTCRequest.from_dict(dict(payload))
    config = TurnTakingConfig.resolve(request.request_data)
    speak_opening_message = (
        isinstance(request.request_data, dict)
        and request.request_data.get("speak_opening_message") is True
    )
    if not config.enabled:
        raise RuntimeError("Realtime turn-taking is disabled by server configuration.")

    async def start_connection(connection: SmallWebRTCConnection) -> None:
        await _cancel_session_pipeline(session_id)

        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=config.audio_input_sample_rate,
                audio_out_sample_rate=config.audio_output_sample_rate,
                audio_out_10ms_chunks=config.output_audio_chunk_ms // 10,
            ),
        )

        task = asyncio.create_task(
            _run_voice_pipeline(
                transport,
                session_id=session_id,
                opening_message=(
                    _opening_message(session) if speak_opening_message else ""
                ),
                config=config,
            ),
            name=f"realtime-voice-{session_id}-{connection.pc_id}",
        )
        _track_task(task, session_id)

    return await _webrtc_handler.handle_web_request(request, start_connection)


async def handle_websocket(
    websocket: Any,
    *,
    session_id: str,
    request_data: dict[str, Any],
    speak_opening_message: bool,
) -> None:
    """Run the same realtime pipeline over a tunnel-friendly WebSocket transport."""
    session = await orchestrator_client.get_session(session_id)
    config = TurnTakingConfig.resolve(request_data)
    if not config.enabled:
        raise RuntimeError("Realtime turn-taking is disabled by server configuration.")

    await _cancel_session_pipeline(session_id)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=config.audio_input_sample_rate,
            audio_out_sample_rate=config.audio_output_sample_rate,
            audio_out_10ms_chunks=config.output_audio_chunk_ms // 10,
            serializer=ProtobufFrameSerializer(),
            add_wav_header=False,
        ),
    )
    task = asyncio.create_task(
        _run_voice_pipeline(
            transport,
            session_id=session_id,
            opening_message=(
                _opening_message(session) if speak_opening_message else ""
            ),
            config=config,
        ),
        name=f"realtime-voice-{session_id}-websocket",
    )
    _track_task(task, session_id)
    await task


async def handle_ice_patch(payload: dict[str, Any]) -> None:
    request = SmallWebRTCPatchRequest(
        pc_id=str(payload.get("pc_id") or ""),
        candidates=[IceCandidate(**candidate) for candidate in payload.get("candidates") or []],
    )
    await _webrtc_handler.handle_patch_request(request)


async def shutdown() -> None:
    tasks = list(_active_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _session_tasks.clear()
    await _webrtc_handler.close()
