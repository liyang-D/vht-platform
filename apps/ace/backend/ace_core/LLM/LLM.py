import os
from dataclasses import dataclass

import httpx


@dataclass
class _Message:
    content: str


class OrchestratorChat:
    def __init__(self, temperature: float, max_tokens: int):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
        self.timeout = float(os.getenv("ACE_ORCHESTRATOR_TIMEOUT_SECONDS", "300"))

    def invoke(self, messages):
        prompt = "\n\n".join(
            f"{getattr(message, 'type', message.__class__.__name__)}: {message.content}"
            for message in messages
        )
        response = httpx.post(
            f"{self.base_url}/inference/text",
            json={
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "priority": 50,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _Message(response.json()["text"])


llm_strict = OrchestratorChat(temperature=0.0, max_tokens=512)
llm_warm = OrchestratorChat(temperature=0.7, max_tokens=256)
