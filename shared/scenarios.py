from typing import TypedDict


class Scenario(TypedDict):
    name: str
    role: str
    instructions: str


DEFAULT_MENTAL_HEALTH_SCENARIO: Scenario = {
    "name": "Patient-facing mental health information",
    "role": (
        "You are a patient-facing mental health information assistant used for "
        "exploratory model comparison."
    ),
    "instructions": (
        "Provide clear and accessible general information while using appropriate "
        "mental health terminology when helpful. Explain technical terms in plain "
        "language.\n\n"
        "Do not diagnose the user or claim certainty about an individual condition. "
        "Distinguish general information from personalised medical advice. If the user "
        "describes an immediate risk of harm to themselves or someone else, advise them "
        "to seek urgent local professional help.\n\n"
        "Keep the conversation natural and respond directly to the user’s question."
    ),
}


def default_scenario() -> Scenario:
    return DEFAULT_MENTAL_HEALTH_SCENARIO.copy()


def resolve_scenario(override: dict[str, object] | None = None) -> Scenario:
    scenario = default_scenario()
    if not override:
        return scenario

    role_value = override.get("role")
    instructions_value = override.get("instructions")
    role = role_value.strip() if isinstance(role_value, str) else ""
    instructions = (
        instructions_value.strip() if isinstance(instructions_value, str) else ""
    )
    if role:
        scenario["role"] = role
    if instructions:
        scenario["instructions"] = instructions
    scenario["name"] = "Custom scenario"
    return scenario
