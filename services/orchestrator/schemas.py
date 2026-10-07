import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.schemas import (
    InteractionMode,
    ResponseModality,
    RuntimeState,
    StructuredStep,
)


class TaskConfig(BaseModel):
    task_name: str = Field(default="Mini Project Template")
    role: str = Field(default="You are a helpful virtual human assistant.")
    instructions: str = Field(default="Answer the user clearly and naturally.")
    language: str | None = None
    requires_structured_output: bool = Field(default=False)
    structured_output_schema: dict[str, Any] | None = None
    lora_adapter: str | None = Field(
        default=None,
        description="LoRA adapter directory name under the configured runtime LoRA root.",
    )


class CreateSessionRequest(BaseModel):
    access_key: str
    task_config: TaskConfig = Field(default_factory=TaskConfig)
    response_modality: ResponseModality = "text"
    synthesize_audio: bool = True
    runtime_state: RuntimeState | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    project_id: str
    model_weights: str
    message: str
    opening_message: str
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class SendMessageRequest(BaseModel):
    text: str
    response_modality: ResponseModality = "text"
    synthesize_audio: bool = True
    interaction_mode: InteractionMode = "free"
    previous_step: StructuredStep | None = None
    current_step: StructuredStep | None = None
    next_step: StructuredStep | None = None


class SendMessageResponse(BaseModel):
    session_id: str
    turn_id: str
    text: str
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class TranscribeAudioResponse(BaseModel):
    session_id: str
    transcript: str
    usage: dict[str, Any] | None = None


class SendAudioMessageResponse(BaseModel):
    session_id: str
    turn_id: str
    transcript: str
    text: str
    audio_base64: str
    audio_mime_type: str
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class SessionMessage(BaseModel):
    id: str
    turn_id: str
    role: str
    text: str
    metadata: dict[str, Any] | None = None
    created_at: str


class SessionTurn(BaseModel):
    id: str
    interaction_mode: str
    status: str
    previous_step: dict[str, Any] | None = None
    current_step: dict[str, Any] | None = None
    next_step: dict[str, Any] | None = None
    structured_output: dict[str, Any] | None = None
    prompt_metadata: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class GetSessionResponse(BaseModel):
    session_id: str
    project_id: str
    model_weights: str
    summary: str | None = None
    task_config: dict[str, Any]
    runtime_state: dict[str, Any] | None = None
    turns: list[SessionTurn] = Field(default_factory=list)
    messages: list[SessionMessage]


class EndSessionResponse(BaseModel):
    session_id: str
    summary: str


class LoraAdapterResponse(BaseModel):
    name: str
    path: str
    base_model: str | None = None
    base_revision: str | None = None
    rank: int | None = None
    peft_type: str | None = None
    weights_file: str | None = None
    loaded: bool | None = None
    valid: bool = True
    error: str | None = None


class LoraAdapterListResponse(BaseModel):
    adapters: list[LoraAdapterResponse]


class LoraAdapterActionResponse(BaseModel):
    adapter: LoraAdapterResponse
    changed: bool


class VisionImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mime_type: str = Field(pattern=r"^image/(jpeg|png|webp)$")
    data_base64: str = Field(min_length=1, max_length=11_200_000)


class VisionAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=32_000)
    images: list[VisionImageInput] = Field(min_length=1, max_length=9)
    max_tokens: int = Field(default=512, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    priority: int = Field(default=100, ge=-1_000_000, le=1_000_000)
    response_schema: dict[str, Any] | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt cannot be blank.")
        return stripped

    @model_validator(mode="after")
    def validate_aggregate_size(self):
        if sum(len(image.data_base64) for image in self.images) > 44_800_000:
            raise ValueError("Combined base64 image payload is too large.")
        if self.response_schema is not None:
            encoded_schema = json.dumps(
                self.response_schema,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(encoded_schema) > 65_536:
                raise ValueError("response_schema is too large.")
        return self


class VisionImageMetadata(BaseModel):
    mime_type: str
    source_width: int
    source_height: int
    width: int
    height: int


class VisionAnalyzeResponse(BaseModel):
    model: str
    text: str
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any]
    images: list[VisionImageMetadata]


class InferenceTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=32_000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=2_000)
    priority: int = Field(default=100, ge=-1_000_000, le=1_000_000)


class InferenceTextResponse(BaseModel):
    model: str
    text: str
    usage: dict[str, Any]


class InferenceSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=8_000)
    priority: int = Field(default=100, ge=-1_000_000, le=1_000_000)


class InferenceSpeechResponse(BaseModel):
    model: str
    audio_base64: str
    audio_mime_type: str
    usage: dict[str, Any]


class InferenceTranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_base64: str = Field(min_length=1, max_length=45_000_000)
    mime_type: str = Field(default="application/octet-stream", max_length=200)
    domain: str | None = Field(default=None, max_length=200)
    boosted_words: list[str] = Field(default_factory=list, max_length=200)
    boost_score: float | None = None
    priority: int = Field(default=100, ge=-1_000_000, le=1_000_000)


class InferenceTranscriptionResponse(BaseModel):
    model: str
    transcript: str
    usage: dict[str, Any]
