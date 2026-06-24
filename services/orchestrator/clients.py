import heapq
import itertools
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI


def load_project_env() -> None:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


load_project_env()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai-compatible")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://llm:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
ORCHESTRATOR_LLM_MAX_CONCURRENT_REQUESTS = int(
    os.getenv("ORCHESTRATOR_LLM_MAX_CONCURRENT_REQUESTS", "4")
)
ORCHESTRATOR_ASR_MAX_CONCURRENT_REQUESTS = int(
    os.getenv("ORCHESTRATOR_ASR_MAX_CONCURRENT_REQUESTS", "2")
)
ORCHESTRATOR_TTS_MAX_CONCURRENT_REQUESTS = int(
    os.getenv("ORCHESTRATOR_TTS_MAX_CONCURRENT_REQUESTS", "1")
)
RUNTIME_REQUEST_MAX_RETRIES = int(os.getenv("RUNTIME_REQUEST_MAX_RETRIES", "2"))
RUNTIME_REQUEST_RETRY_BACKOFF_SECONDS = float(
    os.getenv("RUNTIME_REQUEST_RETRY_BACKOFF_SECONDS", "0.5")
)

DEEPGRAM_API_BASE_URL = os.getenv("DEEPGRAM_API_BASE_URL", "https://api.deepgram.com/v1")
DEEPGRAM_ASR_MODEL = os.getenv("DEEPGRAM_ASR_MODEL", "nova-3")
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
DEEPGRAM_TTS_MIME_TYPE = "audio/wav"

ASR_PROVIDER = os.getenv("ASR_PROVIDER", "riva")
ASR_MODEL = os.getenv("ASR_MODEL", "parakeet-1-1b-ctc-en-us")
ASR_DEFAULT_DOMAIN = os.getenv("ASR_DEFAULT_DOMAIN", "default")
ASR_DEFAULT_MODE = os.getenv("ASR_DEFAULT_MODE", "offline").strip().lower()
ASR_DOMAIN_CONFIG_PATH = os.getenv(
    "ASR_DOMAIN_CONFIG_PATH",
    str(Path(__file__).resolve().parent / "config" / "asr_domains.json"),
)
RIVA_ASR_HTTP_URL = os.getenv("RIVA_ASR_HTTP_URL", "http://asr:9000")
RIVA_ASR_LANGUAGE = os.getenv("RIVA_ASR_LANGUAGE", "en-US")
RIVA_ASR_ENABLE_AUTOMATIC_PUNCTUATION = (
    os.getenv("RIVA_ASR_ENABLE_AUTOMATIC_PUNCTUATION", "true").lower()
    in {"1", "true", "yes", "on"}
)
RIVA_ASR_BOOSTED_WORDS_FIELD = os.getenv(
    "RIVA_ASR_BOOSTED_WORDS_FIELD",
    "boosted_words",
)
RIVA_ASR_BOOST_SCORE_FIELD = os.getenv(
    "RIVA_ASR_BOOST_SCORE_FIELD",
    "boost_score",
)

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "chatterbox")
TTS_MODEL = os.getenv("TTS_MODEL", "chatterbox-turbo")
CHATTERBOX_TTS_HTTP_URL = os.getenv("CHATTERBOX_TTS_HTTP_URL", "http://tts:8000")
CHATTERBOX_TTS_VOICE = os.getenv("CHATTERBOX_TTS_VOICE", "default")
CHATTERBOX_TTS_MIME_TYPE = os.getenv("CHATTERBOX_TTS_MIME_TYPE", "audio/wav")
RIVA_TTS_HTTP_URL = os.getenv("RIVA_TTS_HTTP_URL", "http://tts-riva:9000")
RIVA_TTS_LANGUAGE = os.getenv("RIVA_TTS_LANGUAGE", "en-US")
RIVA_TTS_VOICE = os.getenv("RIVA_TTS_VOICE", "Magpie-Multilingual.EN-US.Aria")
RIVA_TTS_MIME_TYPE = os.getenv("RIVA_TTS_MIME_TYPE", "audio/wav")

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class AsrDomainConfig:
    name: str
    provider: str = ASR_PROVIDER
    model: str = ASR_MODEL
    language: str = RIVA_ASR_LANGUAGE
    supported_modes: tuple[str, ...] = ("offline",)
    offline_http_url: str = RIVA_ASR_HTTP_URL
    streaming_ws_url: str | None = None
    enable_automatic_punctuation: bool = RIVA_ASR_ENABLE_AUTOMATIC_PUNCTUATION
    default_boosted_words: tuple[str, ...] = field(default_factory=tuple)
    boost_score: float | None = None
    extra_form_data: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AsrTranscriptionOptions:
    domain: str | None = None
    mode: str = ASR_DEFAULT_MODE
    boosted_words: tuple[str, ...] = field(default_factory=tuple)
    boost_score: float | None = None
    resolved_domain: AsrDomainConfig | None = None


def parse_boosted_words(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, list | tuple | set):
        candidates = value
    else:
        candidates = [value]

    words: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for raw_word in str(candidate).split(","):
            word = raw_word.strip()
            if not word or word in seen:
                continue

            seen.add(word)
            words.append(word)

    return tuple(words)


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default

    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, list | tuple | set):
        candidates = value
    else:
        candidates = [value]

    parsed = tuple(
        str(candidate).strip().lower()
        for candidate in candidates
        if str(candidate).strip()
    )
    return parsed or default


def parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).lower() in {"1", "true", "yes", "on"}


def default_asr_domain_config(name: str = ASR_DEFAULT_DOMAIN) -> AsrDomainConfig:
    return AsrDomainConfig(name=name)


def load_asr_domain_registry() -> tuple[str, dict[str, AsrDomainConfig]]:
    config_path = Path(ASR_DOMAIN_CONFIG_PATH)
    fallback = default_asr_domain_config()

    if not config_path.exists():
        return ASR_DEFAULT_DOMAIN, {ASR_DEFAULT_DOMAIN: fallback}

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ASR_DEFAULT_DOMAIN, {ASR_DEFAULT_DOMAIN: fallback}

    if not isinstance(raw_config, dict):
        return ASR_DEFAULT_DOMAIN, {ASR_DEFAULT_DOMAIN: fallback}

    default_domain = str(raw_config.get("default_domain") or ASR_DEFAULT_DOMAIN)
    raw_domains = raw_config.get("domains") or {}
    if not isinstance(raw_domains, dict):
        raw_domains = {}

    domains: dict[str, AsrDomainConfig] = {}

    for domain_name, raw_domain in raw_domains.items():
        if not isinstance(raw_domain, dict):
            continue

        name = str(domain_name).strip()
        if not name:
            continue

        extra_form_data = raw_domain.get("extra_form_data") or {}
        if not isinstance(extra_form_data, dict):
            extra_form_data = {}

        domains[name] = AsrDomainConfig(
            name=name,
            provider=str(raw_domain.get("provider") or ASR_PROVIDER),
            model=str(raw_domain.get("model") or ASR_MODEL),
            language=str(raw_domain.get("language") or RIVA_ASR_LANGUAGE),
            supported_modes=parse_string_tuple(
                raw_domain.get("supported_modes"),
                ("offline",),
            ),
            offline_http_url=str(
                raw_domain.get("offline_http_url")
                or raw_domain.get("http_url")
                or RIVA_ASR_HTTP_URL
            ),
            streaming_ws_url=raw_domain.get("streaming_ws_url"),
            enable_automatic_punctuation=parse_bool(
                raw_domain.get("enable_automatic_punctuation"),
                RIVA_ASR_ENABLE_AUTOMATIC_PUNCTUATION,
            ),
            default_boosted_words=parse_boosted_words(
                raw_domain.get("default_boosted_words")
            ),
            boost_score=parse_optional_float(raw_domain.get("boost_score")),
            extra_form_data={str(k): str(v) for k, v in extra_form_data.items()},
        )

    if default_domain not in domains:
        domains[default_domain] = AsrDomainConfig(
            name=default_domain,
            provider=ASR_PROVIDER,
            model=ASR_MODEL,
            language=RIVA_ASR_LANGUAGE,
            supported_modes=("offline",),
            offline_http_url=RIVA_ASR_HTTP_URL,
            enable_automatic_punctuation=RIVA_ASR_ENABLE_AUTOMATIC_PUNCTUATION,
        )

    return default_domain, domains


def resolve_asr_domain(domain: str | None) -> AsrDomainConfig:
    default_domain, domains = load_asr_domain_registry()
    requested_domain = (domain or "").strip()

    if requested_domain and requested_domain in domains:
        return domains[requested_domain]

    return domains.get(default_domain) or default_asr_domain_config(default_domain)


def merge_boosted_words(*word_groups: tuple[str, ...]) -> tuple[str, ...]:
    words: list[str] = []
    seen: set[str] = set()

    for word_group in word_groups:
        for word in word_group:
            if word in seen:
                continue

            seen.add(word)
            words.append(word)

    return tuple(words)


class PriorityGate:
    def __init__(self, max_concurrent_requests: int) -> None:
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self._condition = threading.Condition()
        self._counter = itertools.count()
        self._queue: list[tuple[int, int, object]] = []
        self._active_requests = 0

    @contextmanager
    def acquire(self, priority: int):
        ticket = object()
        sequence = next(self._counter)

        with self._condition:
            heapq.heappush(self._queue, (priority, sequence, ticket))

            while (
                self._active_requests >= self.max_concurrent_requests
                or self._queue[0][2] is not ticket
            ):
                self._condition.wait()

            heapq.heappop(self._queue)
            self._active_requests += 1

        try:
            yield
        finally:
            with self._condition:
                self._active_requests -= 1
                self._condition.notify_all()


LLM_PRIORITY_GATE = PriorityGate(ORCHESTRATOR_LLM_MAX_CONCURRENT_REQUESTS)
ASR_PRIORITY_GATE = PriorityGate(ORCHESTRATOR_ASR_MAX_CONCURRENT_REQUESTS)
TTS_PRIORITY_GATE = PriorityGate(ORCHESTRATOR_TTS_MAX_CONCURRENT_REQUESTS)


def post_with_retries(
    url: str,
    *,
    priority: int,
    timeout: float,
    **kwargs: Any,
) -> httpx.Response:
    last_error: Exception | None = None

    for attempt in range(RUNTIME_REQUEST_MAX_RETRIES + 1):
        try:
            response = httpx.post(url, timeout=timeout, **kwargs)

            if response.status_code not in RETRYABLE_STATUS_CODES:
                return response

            last_error = httpx.HTTPStatusError(
                "Retryable upstream status.",
                request=response.request,
                response=response,
            )
        except httpx.RequestError as e:
            last_error = e

        if attempt < RUNTIME_REQUEST_MAX_RETRIES:
            delay = RUNTIME_REQUEST_RETRY_BACKOFF_SECONDS * (2**attempt)
            delay += min(priority, 1000) / 10000
            time.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError("Request failed before a response was received.")


def get_llm_client() -> OpenAI:
    if LLM_PROVIDER != "openai-compatible":
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    api_key = LLM_API_KEY

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set in .env")

    return OpenAI(api_key=api_key, base_url=LLM_API_BASE_URL)


def get_deepgram_api_key() -> str:
    api_key = os.getenv("DEEPGRAM_API_KEY")

    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")

    return api_key


def extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": 1,
        }

    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)

    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)

    total_tokens = getattr(usage, "total_tokens", None)

    if total_tokens is None:
        total_tokens = 1

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_llm(
    prompt: str,
    priority: int = 100,
) -> tuple[str, dict[str, Any]]:
    client = get_llm_client()

    with LLM_PRIORITY_GATE.acquire(priority):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={
                "priority": priority,
            },
        )

    text = response.choices[0].message.content or ""
    usage = extract_usage(response)

    return text, usage


def summarise_session(
    conversation_text: str,
    priority: int = 100,
) -> tuple[str, dict[str, Any]]:
    prompt = f"""
Summarise the following conversation clearly and concisely.

Focus on:
- user's main needs
- avatar's main responses
- any important decisions or outcomes

Conversation:
{conversation_text}
""".strip()

    return call_llm(prompt, priority=priority)


def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
    priority: int = 100,
    domain: str | None = None,
    mode: str | None = None,
    boosted_words: list[str] | tuple[str, ...] | None = None,
    boost_score: float | None = None,
) -> tuple[str, dict[str, Any]]:
    if not audio_bytes:
        raise RuntimeError("Audio payload is empty.")

    options = AsrTranscriptionOptions(
        domain=domain,
        mode=(mode or ASR_DEFAULT_MODE).strip().lower() or ASR_DEFAULT_MODE,
        boosted_words=parse_boosted_words(boosted_words),
        boost_score=boost_score,
        resolved_domain=resolve_asr_domain(domain),
    )
    provider = options.resolved_domain.provider if options.resolved_domain else ASR_PROVIDER

    if provider == "deepgram":
        return transcribe_audio_deepgram(
            audio_bytes,
            mime_type,
            priority=priority,
            options=options,
        )

    if provider == "riva":
        return transcribe_audio_riva(
            audio_bytes,
            mime_type,
            priority=priority,
            options=options,
        )

    raise RuntimeError(f"Unsupported ASR provider for domain: {provider}")


def transcribe_audio_deepgram(
    audio_bytes: bytes,
    mime_type: str,
    priority: int,
    options: AsrTranscriptionOptions,
) -> tuple[str, dict[str, Any]]:
    domain_config = options.resolved_domain or resolve_asr_domain(options.domain)
    mode = options.mode
    if mode not in domain_config.supported_modes:
        mode = ASR_DEFAULT_MODE

    if mode != "offline":
        raise RuntimeError(
            "Streaming ASR requires a streaming transport; this upload endpoint "
            "currently supports offline transcription only."
        )

    boosted_words = merge_boosted_words(
        domain_config.default_boosted_words,
        options.boosted_words,
    )
    params: list[tuple[str, str]] = [
        ("model", DEEPGRAM_ASR_MODEL),
        ("smart_format", "true"),
    ]

    for word in boosted_words:
        params.append(("keywords", word))

    with ASR_PRIORITY_GATE.acquire(priority):
        response = post_with_retries(
            f"{DEEPGRAM_API_BASE_URL}/listen",
            priority=priority,
            params=params,
            headers={
                "Authorization": f"Token {get_deepgram_api_key()}",
                "Content-Type": mime_type or "application/octet-stream",
            },
            content=audio_bytes,
            timeout=60.0,
        )
    response.raise_for_status()

    payload = response.json()
    alternatives = (
        payload.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [])
    )

    transcript = ""
    confidence = None
    if alternatives:
        transcript = alternatives[0].get("transcript", "")
        confidence = alternatives[0].get("confidence")

    return transcript.strip(), {
        "provider": "deepgram",
        "model": DEEPGRAM_ASR_MODEL,
        "domain": domain_config.name,
        "mode": mode,
        "supported_modes": list(domain_config.supported_modes),
        "boosted_words": list(boosted_words),
        "confidence": confidence,
    }


def parse_riva_transcription(payload: dict[str, Any]) -> tuple[str, float | None]:
    if isinstance(payload.get("text"), str):
        return payload["text"].strip(), payload.get("confidence")

    if isinstance(payload.get("transcript"), str):
        return payload["transcript"].strip(), payload.get("confidence")

    results = payload.get("results")
    if isinstance(results, list) and results:
        alternatives = results[0].get("alternatives", [])
        if alternatives:
            alternative = alternatives[0]
            return (
                str(alternative.get("transcript", "")).strip(),
                alternative.get("confidence"),
            )

    return "", None


def transcribe_audio_riva(
    audio_bytes: bytes,
    mime_type: str,
    priority: int,
    options: AsrTranscriptionOptions,
) -> tuple[str, dict[str, Any]]:
    domain_config = options.resolved_domain or resolve_asr_domain(options.domain)
    mode = options.mode
    if mode not in domain_config.supported_modes:
        mode = ASR_DEFAULT_MODE

    if mode != "offline":
        raise RuntimeError(
            "Streaming ASR requires a streaming transport; this upload endpoint "
            "currently supports offline transcription only."
        )

    filename = "audio"
    if "webm" in mime_type:
        filename = "audio.webm"
    elif "flac" in mime_type:
        filename = "audio.flac"
    elif "wav" in mime_type or "wave" in mime_type:
        filename = "audio.wav"

    boosted_words = merge_boosted_words(
        domain_config.default_boosted_words,
        options.boosted_words,
    )
    boost_score = (
        options.boost_score
        if options.boost_score is not None
        else domain_config.boost_score
    )

    form_data: dict[str, str] = {
        "language": domain_config.language,
        **domain_config.extra_form_data,
    }
    if domain_config.enable_automatic_punctuation:
        form_data["enable_automatic_punctuation"] = "true"

    if boosted_words:
        form_data[RIVA_ASR_BOOSTED_WORDS_FIELD] = ",".join(boosted_words)

    if boosted_words and boost_score is not None:
        form_data[RIVA_ASR_BOOST_SCORE_FIELD] = str(boost_score)

    with ASR_PRIORITY_GATE.acquire(priority):
        response = post_with_retries(
            f"{domain_config.offline_http_url.rstrip('/')}/v1/audio/transcriptions",
            priority=priority,
            data=form_data,
            files={
                "file": (
                    filename,
                    audio_bytes,
                    mime_type or "application/octet-stream",
                )
            },
            timeout=120.0,
        )
    response.raise_for_status()

    transcript, confidence = parse_riva_transcription(response.json())

    return transcript, {
        "provider": "riva",
        "model": domain_config.model,
        "domain": domain_config.name,
        "mode": mode,
        "supported_modes": list(domain_config.supported_modes),
        "language": domain_config.language,
        "boosted_words": list(boosted_words),
        "boost_score": boost_score,
        "confidence": confidence,
    }


def synthesise_speech(
    text: str,
    priority: int = 100,
) -> tuple[bytes, str, dict[str, Any]]:
    if not text.strip():
        raise RuntimeError("Text payload is empty.")

    if TTS_PROVIDER == "deepgram":
        return synthesise_speech_deepgram(text, priority=priority)

    if TTS_PROVIDER == "riva":
        return synthesise_speech_riva(text, priority=priority)

    if TTS_PROVIDER == "chatterbox":
        return synthesise_speech_chatterbox(text, priority=priority)

    raise RuntimeError(f"Unsupported TTS_PROVIDER: {TTS_PROVIDER}")


def synthesise_speech_deepgram(
    text: str,
    priority: int,
) -> tuple[bytes, str, dict[str, Any]]:
    with TTS_PRIORITY_GATE.acquire(priority):
        response = post_with_retries(
            f"{DEEPGRAM_API_BASE_URL}/speak",
            priority=priority,
            params={
                "model": DEEPGRAM_TTS_MODEL,
            },
            headers={
                "Authorization": f"Token {get_deepgram_api_key()}",
                "Content-Type": "application/json",
                "Accept": DEEPGRAM_TTS_MIME_TYPE,
            },
            json={"text": text},
            timeout=60.0,
        )
    response.raise_for_status()

    mime_type = response.headers.get("content-type", DEEPGRAM_TTS_MIME_TYPE)

    return response.content, mime_type, {
        "provider": "deepgram",
        "model": DEEPGRAM_TTS_MODEL,
    }


def synthesise_speech_riva(
    text: str,
    priority: int,
) -> tuple[bytes, str, dict[str, Any]]:
    with TTS_PRIORITY_GATE.acquire(priority):
        response = post_with_retries(
            f"{RIVA_TTS_HTTP_URL.rstrip('/')}/v1/audio/synthesize",
            priority=priority,
            data={
                "language": RIVA_TTS_LANGUAGE,
                "text": text,
                "voice": RIVA_TTS_VOICE,
            },
            timeout=120.0,
        )
    response.raise_for_status()

    mime_type = response.headers.get("content-type", RIVA_TTS_MIME_TYPE)

    return response.content, mime_type, {
        "provider": "riva",
        "model": TTS_MODEL,
        "language": RIVA_TTS_LANGUAGE,
        "voice": RIVA_TTS_VOICE,
    }


def synthesise_speech_chatterbox(
    text: str,
    priority: int,
) -> tuple[bytes, str, dict[str, Any]]:
    with TTS_PRIORITY_GATE.acquire(priority):
        response = post_with_retries(
            f"{CHATTERBOX_TTS_HTTP_URL.rstrip('/')}/v1/audio/synthesize",
            priority=priority,
            json={
                "text": text,
                "voice": CHATTERBOX_TTS_VOICE,
            },
            timeout=120.0,
        )
    response.raise_for_status()

    mime_type = response.headers.get("content-type", CHATTERBOX_TTS_MIME_TYPE)

    return response.content, mime_type, {
        "provider": "chatterbox",
        "model": TTS_MODEL,
        "voice": CHATTERBOX_TTS_VOICE,
    }
