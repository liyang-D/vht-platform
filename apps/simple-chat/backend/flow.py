from typing import Any

from shared.scenarios import resolve_scenario


def build_chat_task_config(
    lora_adapter: str | None = None,
    scenario_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    scenario = resolve_scenario(scenario_override)
    return {
        "task_name": "Simple Chat",
        "role": scenario["role"],
        "instructions": scenario["instructions"],
        "requires_structured_output": False,
        "structured_output_schema": None,
        "lora_adapter": lora_adapter,
    }
