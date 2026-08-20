import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import flow
import main


class CreateSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

    @patch("main.orchestrator_client.create_session", new_callable=AsyncMock)
    def test_passes_selected_adapter_into_task_config(self, create_session) -> None:
        create_session.return_value = {
            "session_id": "new-session",
            "opening_message": "Hello",
        }

        response = self.client.post(
            "/api/sessions",
            json={"response_modality": "text", "lora_adapter": "pilot-best"},
        )

        self.assertEqual(response.status_code, 200)
        create_session.assert_awaited_once_with(
            task_config=flow.build_chat_task_config("pilot-best"),
            response_modality="text",
        )
        self.assertEqual(
            response.cookies.get(main.SESSION_COOKIE_NAME),
            "new-session",
        )

    @patch("main.orchestrator_client.create_session", new_callable=AsyncMock)
    @patch("main.orchestrator_client.get_session", new_callable=AsyncMock)
    def test_rejects_replacement_while_chat_is_active(
        self,
        get_session,
        create_session,
    ) -> None:
        get_session.return_value = {"summary": None}
        self.client.cookies.set(main.SESSION_COOKIE_NAME, "active-session")

        response = self.client.post(
            "/api/sessions",
            json={"response_modality": "text", "lora_adapter": None},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"],
            "End the active chat before starting a new one.",
        )
        create_session.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
