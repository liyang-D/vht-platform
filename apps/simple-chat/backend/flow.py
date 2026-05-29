from typing import Any


def build_chat_task_config() -> dict[str, Any]:
    return {
        "task_name": "Simple Chat",
        "role": "You are a helpful virtual human assistant in a simple chat app.",
        "instructions": (
            "Respond naturally and concisely. Ask clarifying questions when useful. "
            "Keep the conversation focused on helping the user."
        ),
        "requires_structured_output": False,
        "structured_output_schema": None,
    }
