import json
import re
from typing import Any

from . import clients
from . import db
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


def create_new_session(
    access_key: str,
    task_config: TaskConfig,
) -> dict[str, Any]:
    validate_task_config(task_config)

    access_record = db.get_access_key_with_project(access_key)
    validate_access_record(access_record)

    session_record = db.create_session_record(
        project_id=str(access_record["project_id"]),
        access_key_id=str(access_record["access_key_id"]),
        task_config=task_config.model_dump(),
    )

    task_config_dict = task_config.model_dump()
    opening_prompt = build_opening_prompt(task_config_dict)

    raw_llm_text, usage = clients.call_llm(opening_prompt)
    opening_text, structured_output = parse_llm_output(raw_llm_text, task_config_dict)

    db.add_message(
        session_id=str(session_record["id"]),
        role="avatar",
        text=opening_text,
        structured_output=structured_output,
    )

    usage_amount = usage.get("total_tokens") or 1

    db.increment_usage(
        project_id=str(access_record["project_id"]),
        access_key_id=str(access_record["access_key_id"]),
        amount=int(usage_amount),
    )

    return {
        "session_id": str(session_record["id"]),
        "project_id": str(session_record["project_id"]),
        "message": "Session created successfully.",
        "opening_message": opening_text,
        "structured_output": structured_output,
        "usage": usage,
    }


def format_history(messages: list[dict[str, Any]], max_messages: int = 12) -> str:
    recent_messages = messages[-max_messages:]

    lines = []
    for message in recent_messages:
        role = message["role"]
        text = message["text"]
        lines.append(f"{role}: {text}")

    return "\n".join(lines)


def build_opening_prompt(task_config: dict[str, Any]) -> str:
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


def build_prompt(
    task_config: dict[str, Any],
    summary: str | None,
    history: str,
    latest_user_text: str,
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


def handle_user_message(
    session_id: str,
    user_text: str,
) -> dict[str, Any]:
    session_record = db.get_session_with_project_and_key(session_id)
    validate_session_record(session_record)

    messages = db.get_session_messages(session_id)
    task_config = db.serialise_task_config(session_record.get("task_config"))
    history = format_history(messages)
    prompt = build_prompt(
        task_config=task_config,
        summary=session_record.get("summary"),
        history=history,
        latest_user_text=user_text,
    )

    db.add_message(
        session_id=session_id,
        role="user",
        text=user_text,
    )

    raw_llm_text, usage = clients.call_llm(prompt)
    avatar_text, structured_output = parse_llm_output(raw_llm_text, task_config)

    db.add_message(
        session_id=session_id,
        role="avatar",
        text=avatar_text,
        structured_output=structured_output,
    )

    usage_amount = usage.get("total_tokens") or 1

    db.increment_usage(
        project_id=str(session_record["project_id"]),
        access_key_id=str(session_record["access_key_id"]),
        amount=int(usage_amount),
    )

    return {
        "session_id": session_id,
        "text": avatar_text,
        "structured_output": structured_output,
        "usage": usage,
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