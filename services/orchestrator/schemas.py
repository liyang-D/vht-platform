from typing import Any

from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    task_name: str = Field(default="Mini Project Template")
    role: str = Field(default="You are a helpful virtual human assistant.")
    instructions: str = Field(default="Answer the user clearly and naturally.")
    requires_structured_output: bool = Field(default=False)
    structured_output_schema: dict[str, Any] | None = None


class CreateSessionRequest(BaseModel):
    access_key: str
    task_config: TaskConfig = Field(default_factory=TaskConfig)


class CreateSessionResponse(BaseModel):
    session_id: str
    project_id: str
    message: str
    opening_message: str
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class SendMessageRequest(BaseModel):
    text: str


class SendMessageResponse(BaseModel):
    session_id: str
    text: str
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class SendAudioMessageResponse(BaseModel):
    session_id: str
    transcript: str
    text: str
    audio_base64: str
    audio_mime_type: str
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class SessionMessage(BaseModel):
    id: str
    role: str
    text: str
    structured_output: dict[str, Any] | None = None
    created_at: str


class GetSessionResponse(BaseModel):
    session_id: str
    project_id: str
    summary: str | None = None
    task_config: dict[str, Any]
    messages: list[SessionMessage]


class EndSessionResponse(BaseModel):
    session_id: str
    summary: str


class ErrorResponse(BaseModel):
    detail: str
