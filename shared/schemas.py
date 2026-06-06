from typing import Any, Literal

from pydantic import BaseModel, Field


InteractionMode = Literal["opening", "free", "structured"]
ResponseModality = Literal["text", "voice"]
TurnStatus = Literal["pending", "processing", "completed", "cancelled", "failed"]
MessageRole = Literal["user", "avatar", "system"]
StructuredDecision = Literal[
    "revise_previous",
    "stay_current",
    "advance_next",
]


class StructuredField(BaseModel):
    schema_version: str = "structured_field.v1"
    type: str
    required: bool = True
    value: Any | None = None
    description: str | None = None
    validation: dict[str, Any] | None = None


class StructuredStep(BaseModel):
    schema_version: str = "structured_step.v1"
    step_id: str
    question: str
    requires_structured_output: bool = True
    fields: dict[str, StructuredField] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None


class StructuredState(BaseModel):
    schema_version: str = "structured_state.v1"
    previous_step: StructuredStep | None = None
    current_step: StructuredStep | None = None
    next_step: StructuredStep | None = None


class RuntimeState(BaseModel):
    schema_version: str = "runtime_state.v1"
    interaction_mode: InteractionMode = "free"
    response_modality: ResponseModality = "text"
    structured_state: StructuredState | None = None


class StructuredTurnOutput(BaseModel):
    schema_version: str = "structured_turn_output.v1"
    decision: StructuredDecision
    target_step_id: str
    updated_fields: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    assistant_message: str


class TurnPromptMetadata(BaseModel):
    schema_version: str = "turn_prompt_metadata.v1"
    response_modality: ResponseModality = "text"
    prompt_template: str
    model: str | None = None
    notes: dict[str, Any] | None = None


def model_dump_jsonb(model: BaseModel | None) -> dict[str, Any] | None:
    if model is None:
        return None

    return model.model_dump(mode="json", exclude_none=True)
