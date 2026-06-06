import json
import re
import base64
from typing import Any

from pydantic import BaseModel

from shared.schemas import RuntimeState, StructuredStep, TurnPromptMetadata

from . import clients
from . import db
from . import safety
from .schemas import TaskConfig


class OrchestratorError(Exception):
    pass


class NotFoundError(OrchestratorError):
    pass


class AccessDeniedError(OrchestratorError):
    pass


class QuotaExceededError(OrchestratorError):
    pass


def validate_task_config(task_config: TaskConfig) -> None:
    if task_config.requires_structured_output and not task_config.structured_output_schema:
        raise OrchestratorError(
            "structured_output_schema is required when requires_structured_output is true."
        )


def dump_model(value: BaseModel | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)

    return value


def serialise_json(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        return json.loads(value)

    return dict(value)


def build_runtime_state_payload(
    interaction_mode: str,
    response_modality: str,
    previous_step: dict[str, Any] | None = None,
    current_step: dict[str, Any] | None = None,
    next_step: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structured_state = None
    if interaction_mode == "structured":
        structured_state = {
            "schema_version": "structured_state.v1",
            "previous_step": previous_step,
            "current_step": current_step,
            "next_step": next_step,
        }

    return {
        "schema_version": "runtime_state.v1",
        "interaction_mode": interaction_mode,
        "response_modality": response_modality,
        "structured_state": structured_state,
    }


def check_quota(quota: int | None, usage_count: int) -> bool:
    if quota is None:
        return True

    return usage_count < quota


def validate_access_record(record: dict[str, Any] | None) -> None:
    if record is None:
        raise AccessDeniedError("Invalid access key.")

    if not record["is_active"]:
        raise AccessDeniedError("Access key is inactive.")

    if not record["project_is_valid"]:
        raise AccessDeniedError("Project is not valid.")

    if not check_quota(record["project_quota"], record["project_usage_count"]):
        raise QuotaExceededError("Project quota has been exceeded.")

    if not check_quota(record["access_key_quota"], record["access_key_usage_count"]):
        raise QuotaExceededError("Access key quota has been exceeded.")


def validate_session_record(record: dict[str, Any] | None) -> None:
    if record is None:
        raise NotFoundError("Session not found.")

    if not record["project_is_valid"]:
        raise AccessDeniedError("Project is not valid.")

    if not record["access_key_is_active"]:
        raise AccessDeniedError("Access key is inactive.")

    if not check_quota(record["project_quota"], record["project_usage_count"]):
        raise QuotaExceededError("Project quota has been exceeded.")

    if not check_quota(record["access_key_quota"], record["access_key_usage_count"]):
        raise QuotaExceededError("Access key quota has been exceeded.")


def add_modality_instructions(parts: list[str], response_modality: str) -> None:
    if response_modality != "voice":
        return

    parts.extend(
        [
            "Current interaction mode:",
            "Voice conversation.",
            "",
            "Voice response style:",
            "Write for speech, not for reading on a screen.",
            "Use natural, conversational sentences.",
            "Avoid markdown, bullets, tables, symbols, dense punctuation, code-like formatting, and hard-to-pronounce abbreviations.",
            "Spell out acronyms, units, and shorthand when it helps pronunciation.",
            "Prefer short responses unless the user asks for detail.",
            "",
        ]
    )


def add_response_audio(
    payload: dict[str, Any],
    text: str,
    usage: dict[str, Any],
    response_modality: str,
) -> None:
    if response_modality != "voice":
        return

    audio_bytes, audio_mime_type, speech_usage = clients.synthesise_speech(text)
    payload["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
    payload["audio_mime_type"] = audio_mime_type
    usage["speech"] = speech_usage


def build_opening_prompt(
    task_config: dict[str, Any],
    response_modality: str = "text",
) -> str:
    role = task_config.get("role") or "You are a helpful avatar assistant."
    instructions = task_config.get("instructions") or "Answer clearly and naturally."
    requires_structured_output = task_config.get("requires_structured_output", False)
    structured_output_schema = task_config.get("structured_output_schema")

    if requires_structured_output and not structured_output_schema:
        raise OrchestratorError(
            "structured_output_schema is required when requires_structured_output is true."
        )

    parts = [
        "You are running inside the VHT orchestrator.",
        "",
        "Task role:",
        role,
        "",
        "Task instructions:",
        instructions,
        "",
        "The session has just started.",
        "Generate a brief opening message to greet the user and explain how you can help in this task.",
        "Do not ask too many questions at once.",
        "",
    ]

    add_modality_instructions(parts, response_modality)

    if requires_structured_output:
        parts.extend(
            [
                "Return your answer as valid JSON only.",
                "Use this format:",
                json.dumps(
                    {
                        "text": "Natural language opening message to the user.",
                        "structured_output": structured_output_schema or {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "Do not include markdown fences.",
            ]
        )
    else:
        parts.append("Return only the natural language opening message.")

    return "\n".join(parts).strip()


def create_new_session(
    access_key: str,
    task_config: TaskConfig,
    response_modality: str = "text",
    runtime_state: RuntimeState | None = None,
) -> dict[str, Any]:
    validate_task_config(task_config)

    access_record = db.get_access_key_with_project(access_key)
    validate_access_record(access_record)

    runtime_state_payload = (
        runtime_state.model_dump(mode="json", exclude_none=True)
        if runtime_state is not None
        else build_runtime_state_payload(
            interaction_mode="free",
            response_modality=response_modality,
        )
    )

    session_record = db.create_session_record(
        project_id=str(access_record["project_id"]),
        access_key_id=str(access_record["access_key_id"]),
        task_config=task_config.model_dump(),
        runtime_state=runtime_state_payload,
    )

    task_config_dict = task_config.model_dump()
    opening_prompt = build_opening_prompt(task_config_dict, response_modality)
    prompt_metadata = TurnPromptMetadata(
        response_modality=response_modality,
        prompt_template="opening.v1",
        model=clients.OPENAI_MODEL,
    ).model_dump(mode="json", exclude_none=True)
    turn_record = db.create_turn(
        session_id=str(session_record["id"]),
        interaction_mode="opening",
        status="processing",
        prompt_metadata=prompt_metadata,
    )

    raw_llm_text, usage = clients.call_llm(opening_prompt)
    opening_text, structured_output = parse_llm_output(raw_llm_text, task_config_dict)

    db.add_message(
        turn_id=str(turn_record["id"]),
        role="avatar",
        text=opening_text,
    )
    db.update_turn(
        turn_id=str(turn_record["id"]),
        status="completed",
        structured_output=structured_output,
        prompt_metadata=prompt_metadata,
        usage=usage,
    )

    usage_amount = usage.get("total_tokens") or 1

    db.increment_usage(
        project_id=str(access_record["project_id"]),
        access_key_id=str(access_record["access_key_id"]),
        amount=int(usage_amount),
    )

    payload = {
        "session_id": str(session_record["id"]),
        "project_id": str(session_record["project_id"]),
        "message": "Session created successfully.",
        "opening_message": opening_text,
        "audio_base64": None,
        "audio_mime_type": None,
        "structured_output": structured_output,
        "usage": usage,
    }

    add_response_audio(payload, opening_text, usage, response_modality)
    db.update_turn(
        turn_id=str(turn_record["id"]),
        status="completed",
        structured_output=structured_output,
        prompt_metadata=prompt_metadata,
        usage=usage,
    )

    return payload


def format_history(messages: list[dict[str, Any]], max_messages: int = 12) -> str:
    recent_messages = messages[-max_messages:]

    lines = []
    for message in recent_messages:
        role = message["role"]
        text = message["text"]
        lines.append(f"{role}: {text}")

    return "\n".join(lines)


def build_prompt(
    task_config: dict[str, Any],
    summary: str | None,
    history: str,
    latest_user_text: str,
    response_modality: str = "text",
) -> str:
    role = task_config.get("role") or "You are a helpful avatar assistant."
    instructions = task_config.get("instructions") or "Answer clearly and naturally."
    requires_structured_output = task_config.get("requires_structured_output", False)
    structured_output_schema = task_config.get("structured_output_schema")

    if requires_structured_output and not structured_output_schema:
        raise OrchestratorError(
            "structured_output_schema is required when requires_structured_output is true."
        )

    parts = [
        "You are running inside the VHT orchestrator.",
        "",
        "Task role:",
        role,
        "",
        "Task instructions:",
        instructions,
        "",
    ]

    add_modality_instructions(parts, response_modality)

    if summary:
        parts.extend(
            [
                "Existing session summary:",
                summary,
                "",
            ]
        )

    if history:
        parts.extend(
            [
                "Recent conversation history:",
                history,
                "",
            ]
        )

    parts.extend(
        [
            "Latest user message:",
            latest_user_text,
            "",
        ]
    )

    if requires_structured_output:
        parts.extend(
            [
                "Return your answer as valid JSON only.",
                "Use this format:",
                json.dumps(
                    {
                        "text": "Natural language response to the user.",
                        "structured_output": structured_output_schema or {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "Do not include markdown fences.",
            ]
        )
    else:
        parts.append("Return only the natural language response text.")

    return "\n".join(parts).strip()


def step_requires_structured_output(current_step: dict[str, Any] | None) -> bool:
    if not current_step:
        return False

    return bool(current_step.get("requires_structured_output", True))


def build_structured_prompt(
    task_config: dict[str, Any],
    summary: str | None,
    history: str,
    latest_user_text: str,
    previous_step: dict[str, Any] | None,
    current_step: dict[str, Any] | None,
    next_step: dict[str, Any] | None,
    response_modality: str = "text",
) -> str:
    role = task_config.get("role") or "You are a helpful avatar assistant."
    instructions = task_config.get("instructions") or "Answer clearly and naturally."
    requires_output = step_requires_structured_output(current_step)

    parts = [
        "You are running inside the VHT orchestrator.",
        "",
        "Task role:",
        role,
        "",
        "Task instructions:",
        instructions,
        "",
        "Current interaction mode:",
        "Structured information collection.",
        "",
        "You only have access to the previous, current, and next collection steps.",
        "Use the latest user message to update the previous step if the user is correcting it, otherwise update the current step.",
        "If the current step is complete and valid, move forward to the next step and ask the next question.",
        "If the current step is incomplete or unclear, stay on the current step and ask a focused follow-up.",
        "",
    ]

    add_modality_instructions(parts, response_modality)

    if summary:
        parts.extend(["Existing session summary:", summary, ""])

    if history:
        parts.extend(["Recent conversation history:", history, ""])

    parts.extend(
        [
            "Previous step JSON:",
            json.dumps(previous_step, ensure_ascii=False, indent=2),
            "",
            "Current step JSON:",
            json.dumps(current_step, ensure_ascii=False, indent=2),
            "",
            "Next step JSON:",
            json.dumps(next_step, ensure_ascii=False, indent=2),
            "",
            "Latest user message:",
            latest_user_text,
            "",
        ]
    )

    if not requires_output:
        parts.append("Return only the natural language response text.")
        return "\n".join(parts).strip()

    parts.extend(
        [
            "Return your answer as valid JSON only.",
            "Use exactly this shape:",
            json.dumps(
                {
                    "assistant_message": "Natural language message to the user.",
                    "decision": "revise_previous | stay_current | advance_next",
                    "target_step_id": "The step_id you updated or stayed on.",
                    "updated_fields": {
                        "field_name": "value extracted or corrected from the user"
                    },
                    "missing_fields": ["field_name"],
                    "validation_errors": ["Short validation issue if any"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Do not include markdown fences.",
        ]
    )

    return "\n".join(parts).strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_llm_output(
    raw_text: str,
    task_config: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    requires_structured_output = task_config.get("requires_structured_output", False)

    if not requires_structured_output:
        return raw_text.strip(), None

    parsed = extract_json_object(raw_text)

    if parsed is None:
        return raw_text.strip(), {
            "parse_error": True,
            "raw_output": raw_text,
        }

    avatar_text = (
        parsed.get("text")
        or parsed.get("response")
        or parsed.get("answer")
        or raw_text
    )

    structured_output = parsed.get("structured_output")

    if structured_output is None:
        structured_output = {
            key: value
            for key, value in parsed.items()
            if key not in {"text", "response", "answer"}
        }

    return str(avatar_text).strip(), structured_output


def parse_structured_llm_output(
    raw_text: str,
    requires_structured_output: bool,
) -> tuple[str, dict[str, Any] | None]:
    if not requires_structured_output:
        return raw_text.strip(), None

    parsed = extract_json_object(raw_text)

    if parsed is None:
        return raw_text.strip(), {
            "schema_version": "structured_turn_output.v1",
            "parse_error": True,
            "raw_output": raw_text,
        }

    assistant_message = (
        parsed.get("assistant_message")
        or parsed.get("text")
        or parsed.get("response")
        or parsed.get("answer")
        or raw_text
    )

    structured_output = {
        "schema_version": parsed.get("schema_version", "structured_turn_output.v1"),
        "decision": parsed.get("decision"),
        "target_step_id": parsed.get("target_step_id"),
        "updated_fields": parsed.get("updated_fields") or {},
        "missing_fields": parsed.get("missing_fields") or [],
        "validation_errors": parsed.get("validation_errors") or [],
        "assistant_message": str(assistant_message).strip(),
    }

    return str(assistant_message).strip(), structured_output


def resolve_structured_steps(
    session_record: dict[str, Any],
    previous_step: StructuredStep | None,
    current_step: StructuredStep | None,
    next_step: StructuredStep | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    previous_payload = dump_model(previous_step)
    current_payload = dump_model(current_step)
    next_payload = dump_model(next_step)

    if previous_payload or current_payload or next_payload:
        return previous_payload, current_payload, next_payload

    runtime_state = serialise_json(session_record.get("runtime_state"))
    structured_state = runtime_state.get("structured_state") if runtime_state else None

    if not structured_state:
        return None, None, None

    return (
        structured_state.get("previous_step"),
        structured_state.get("current_step"),
        structured_state.get("next_step"),
    )


def handle_user_message(
    session_id: str,
    user_text: str,
    response_modality: str = "text",
    interaction_mode: str = "free",
    previous_step: StructuredStep | None = None,
    current_step: StructuredStep | None = None,
    next_step: StructuredStep | None = None,
) -> dict[str, Any]:
    session_record = db.get_session_with_project_and_key(session_id)
    validate_session_record(session_record)

    task_config = db.serialise_task_config(session_record.get("task_config"))
    previous_payload = None
    current_payload = None
    next_payload = None

    if interaction_mode == "structured":
        previous_payload, current_payload, next_payload = resolve_structured_steps(
            session_record=session_record,
            previous_step=previous_step,
            current_step=current_step,
            next_step=next_step,
        )
        if current_payload is None:
            raise OrchestratorError(
                "current_step is required for structured interaction mode."
            )

    prompt_metadata = TurnPromptMetadata(
        response_modality=response_modality,
        prompt_template="structured.v1"
        if interaction_mode == "structured"
        else "free.v1",
        model=clients.OPENAI_MODEL,
    ).model_dump(mode="json", exclude_none=True)
    turn_record = db.create_turn(
        session_id=session_id,
        interaction_mode=interaction_mode,
        status="processing",
        previous_step=previous_payload,
        current_step=current_payload,
        next_step=next_payload,
        prompt_metadata=prompt_metadata,
    )
    turn_id = str(turn_record["id"])

    safety_result = safety.basic_input_safety_check(
        user_text=user_text,
        task_config=task_config,
    )

    if not safety_result.allowed:
        refusal_text = safety_result.message or "I’m not able to help with that request."

        db.add_message(
            turn_id=turn_id,
            role="user",
            text=user_text,
        )

        structured_output = {
            "safety_blocked": True,
            "reason": safety_result.reason,
        }
        db.add_message(
            turn_id=turn_id,
            role="avatar",
            text=refusal_text,
        )
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        db.update_turn(
            turn_id=turn_id,
            status="completed",
            structured_output=structured_output,
            prompt_metadata=prompt_metadata,
            usage=usage,
        )
        db.update_session_runtime_state(
            session_id=session_id,
            runtime_state=build_runtime_state_payload(
                interaction_mode=interaction_mode,
                response_modality=response_modality,
                previous_step=previous_payload,
                current_step=current_payload,
                next_step=next_payload,
            ),
        )

        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "text": refusal_text,
            "structured_output": structured_output,
            "usage": usage,
        }

    messages = db.get_session_messages(session_id)
    db.add_message(
        turn_id=turn_id,
        role="user",
        text=user_text,
    )

    history = format_history(messages)
    if interaction_mode == "structured":
        prompt = build_structured_prompt(
            task_config=task_config,
            summary=session_record.get("summary"),
            history=history,
            latest_user_text=user_text,
            previous_step=previous_payload,
            current_step=current_payload,
            next_step=next_payload,
            response_modality=response_modality,
        )
        requires_output = step_requires_structured_output(current_payload)
    else:
        prompt = build_prompt(
            task_config=task_config,
            summary=session_record.get("summary"),
            history=history,
            latest_user_text=user_text,
            response_modality=response_modality,
        )
        requires_output = task_config.get("requires_structured_output", False)

    raw_llm_text, usage = clients.call_llm(prompt)
    if interaction_mode == "structured":
        avatar_text, structured_output = parse_structured_llm_output(
            raw_text=raw_llm_text,
            requires_structured_output=requires_output,
        )
    else:
        avatar_text, structured_output = parse_llm_output(raw_llm_text, task_config)

    db.add_message(
        turn_id=turn_id,
        role="avatar",
        text=avatar_text,
    )
    db.update_turn(
        turn_id=turn_id,
        status="completed",
        structured_output=structured_output,
        prompt_metadata=prompt_metadata,
        usage=usage,
    )

    db.update_session_runtime_state(
        session_id=session_id,
        runtime_state=build_runtime_state_payload(
            interaction_mode=interaction_mode,
            response_modality=response_modality,
            previous_step=previous_payload,
            current_step=current_payload,
            next_step=next_payload,
        ),
    )

    usage_amount = usage.get("total_tokens") or 1

    db.increment_usage(
        project_id=str(session_record["project_id"]),
        access_key_id=str(session_record["access_key_id"]),
        amount=int(usage_amount),
    )

    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "text": avatar_text,
        "audio_base64": None,
        "audio_mime_type": None,
        "structured_output": structured_output,
        "usage": usage,
    }

    add_response_audio(payload, avatar_text, usage, response_modality)
    db.update_turn(
        turn_id=turn_id,
        status="completed",
        structured_output=structured_output,
        prompt_metadata=prompt_metadata,
        usage=usage,
    )

    return payload


def transcribe_user_audio_message(
    session_id: str,
    audio_bytes: bytes,
    mime_type: str,
) -> dict[str, Any]:
    session_record = db.get_session_with_project_and_key(session_id)
    validate_session_record(session_record)

    transcript, transcription_usage = clients.transcribe_audio(
        audio_bytes=audio_bytes,
        mime_type=mime_type,
    )

    if not transcript:
        raise OrchestratorError("No speech was detected in the audio.")

    return {
        "session_id": session_id,
        "transcript": transcript,
        "usage": {
            "transcription": transcription_usage,
        },
    }


def handle_user_audio_message(
    session_id: str,
    audio_bytes: bytes,
    mime_type: str,
) -> dict[str, Any]:
    transcription = transcribe_user_audio_message(
        session_id=session_id,
        audio_bytes=audio_bytes,
        mime_type=mime_type,
    )
    transcript = transcription["transcript"]

    response = handle_user_message(
        session_id=session_id,
        user_text=transcript,
        response_modality="voice",
    )

    usage = response.get("usage") or {}
    usage["transcription"] = transcription["usage"]["transcription"]

    return {
        "session_id": session_id,
        "turn_id": response["turn_id"],
        "transcript": transcript,
        "text": response["text"],
        "audio_base64": response["audio_base64"],
        "audio_mime_type": response["audio_mime_type"],
        "structured_output": response.get("structured_output"),
        "usage": usage,
    }


def get_existing_session(session_id: str) -> dict[str, Any]:
    session_record = db.get_session_with_project_and_key(session_id)
    validate_session_record(session_record)

    messages = db.get_session_messages(session_id)
    turns = db.get_session_turns(session_id)

    return {
        "session_id": session_id,
        "project_id": str(session_record["project_id"]),
        "summary": session_record.get("summary"),
        "task_config": db.serialise_task_config(session_record.get("task_config")),
        "runtime_state": serialise_json(session_record.get("runtime_state")),
        "turns": [
            {
                "id": str(turn["id"]),
                "interaction_mode": turn["interaction_mode"],
                "status": turn["status"],
                "previous_step": turn.get("previous_step"),
                "current_step": turn.get("current_step"),
                "next_step": turn.get("next_step"),
                "structured_output": turn.get("structured_output"),
                "prompt_metadata": turn.get("prompt_metadata"),
                "usage": turn.get("usage"),
                "created_at": turn["created_at"].isoformat(),
                "started_at": turn["started_at"].isoformat()
                if turn.get("started_at")
                else None,
                "completed_at": turn["completed_at"].isoformat()
                if turn.get("completed_at")
                else None,
            }
            for turn in turns
        ],
        "messages": [
            {
                "id": str(message["id"]),
                "turn_id": str(message["turn_id"]),
                "role": message["role"],
                "text": message["text"],
                "metadata": message.get("metadata"),
                "created_at": message["created_at"].isoformat(),
            }
            for message in messages
        ],
    }


def build_conversation_text(messages: list[dict[str, Any]]) -> str:
    lines = []

    for message in messages:
        role = message["role"]
        text = message["text"]
        lines.append(f"{role}: {text}")

    return "\n".join(lines)


def end_existing_session(session_id: str) -> dict[str, Any]:
    session_record = db.get_session_with_project_and_key(session_id)

    if session_record is None:
        raise NotFoundError("Session not found.")

    messages = db.get_session_messages(session_id)

    if not messages:
        summary = "No messages were recorded in this session."
    else:
        conversation_text = build_conversation_text(messages)
        summary, usage = clients.summarise_session(conversation_text)

        usage_amount = usage.get("total_tokens") or 1
        db.increment_usage(
            project_id=str(session_record["project_id"]),
            access_key_id=str(session_record["access_key_id"]),
            amount=int(usage_amount),
        )

    db.update_session_summary(session_id, summary)

    return {
        "session_id": session_id,
        "summary": summary,
    }
