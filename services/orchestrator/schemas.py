from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

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


class LLMOptions(BaseModel):
    thinking_mode: Literal["default", "think", "no_think"] = "default"


class CreateSessionRequest(BaseModel):
    access_key: str
    task_config: TaskConfig = Field(default_factory=TaskConfig)
    response_modality: ResponseModality = "text"
    runtime_state: RuntimeState | None = None
    llm_options: LLMOptions = Field(default_factory=LLMOptions)


class CreateSessionResponse(BaseModel):
    session_id: str
    project_id: str
    message: str
    opening_message: str
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class SendMessageRequest(BaseModel):
    text: str
    response_modality: ResponseModality = "text"
    interaction_mode: InteractionMode = "free"
    previous_step: StructuredStep | None = None
    current_step: StructuredStep | None = None
    next_step: StructuredStep | None = None
    llm_options: LLMOptions = Field(default_factory=LLMOptions)


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
    summary: str | None = None
    task_config: dict[str, Any]
    runtime_state: dict[str, Any] | None = None
    turns: list[SessionTurn] = Field(default_factory=list)
    messages: list[SessionMessage]


class EndSessionResponse(BaseModel):
    session_id: str
    summary: str


class ErrorResponse(BaseModel):
    detail: str
