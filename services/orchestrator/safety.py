from dataclasses import dataclass
from typing import Literal


SafetyReason = Literal[
    "allowed",
    "empty_input",
    "input_too_long",
    "out_of_scope",
    "unsafe_content",
]


@dataclass
class SafetyResult:
    allowed: bool
    reason: SafetyReason
    message: str | None = None


OUT_OF_SCOPE_KEYWORDS = {
    "bitcoin",
    "crypto",
    "stock market",
    "investment advice",
    "forex",
    "casino",
    "gambling",
}

UNSAFE_KEYWORDS = {
    "make a bomb",
    "build a bomb",
    "weapon",
    "poison",
    "kill someone",
}


def basic_input_safety_check(
    user_text: str,
    task_config: dict,
) -> SafetyResult:
    text = user_text.strip()
    lower_text = text.lower()

    if not text:
        return SafetyResult(
            allowed=False,
            reason="empty_input",
            message="Please enter a message so I can help with this task.",
        )

    if len(text) > 4000:
        return SafetyResult(
            allowed=False,
            reason="input_too_long",
            message="That message is too long for this prototype. Please shorten it and try again.",
        )

    for keyword in UNSAFE_KEYWORDS:
        if keyword in lower_text:
            return SafetyResult(
                allowed=False,
                reason="unsafe_content",
                message="I’m not able to help with that. Please keep the conversation focused on the research task.",
            )

    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in lower_text:
            return SafetyResult(
                allowed=False,
                reason="out_of_scope",
                message="I’m only able to help with this research task, so I can’t answer questions outside this scope.",
            )

    return SafetyResult(
        allowed=True,
        reason="allowed",
        message=None,
    )