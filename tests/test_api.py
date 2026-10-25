import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

from fastapi import HTTPException
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import (
    AnswerResponse,
    QuestionRequest,
    app,
    dashboard,
    health,
    submit_question,
)


class SupportApiTests(unittest.TestCase):
    def test_chat_endpoint_returns_mocked_answer_and_policy_sources(self):
        service = Mock(return_value={
            "answer": "You can request a return within 7 calendar days of delivery.",
            "status": "answered",
            "sources": [{
                "source": "refund_policy.txt",
                "section": "1. Return Window",
                "chunk_index": 0,
                "id": "refund_policy-0",
                "distance": 0.26,
            }],
            "coverage": {"status": "recognized_policy_topic", "topics": ["returns"]},
        })
        response = submit_question(
            QuestionRequest(question="How many days do I have to return an order?"),
            service=service,
        )

        validated = AnswerResponse.model_validate(response)
        self.assertEqual(validated.status, "answered")
        self.assertEqual(validated.sources[0].source, "refund_policy.txt")
        self.assertEqual(validated.sources[0].section, "1. Return Window")
        service.assert_called_once_with("How many days do I have to return an order?")

    def test_chat_endpoint_validates_empty_and_oversized_questions(self):
        service = Mock()
        with self.assertRaises(ValidationError):
            QuestionRequest(question="   ")
        with self.assertRaises(ValidationError):
            QuestionRequest(question="x" * 1001)
        with self.assertRaises(ValidationError):
            QuestionRequest(question=123)
        service.assert_not_called()

    def test_chat_endpoint_returns_useful_service_error(self):
        service = Mock(side_effect=RuntimeError("provider internals should not leak"))
        with self.assertLogs("app.main", level="ERROR"):
            with self.assertRaises(HTTPException) as raised:
                submit_question(QuestionRequest(question="How long is standard delivery?"), service=service)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            "The support service is temporarily unavailable. Please try again.",
        )
        self.assertNotIn("provider internals", raised.exception.detail)

    def test_dashboard_and_static_assets_are_served(self):
        root_response = dashboard()
        self.assertTrue(Path(root_response.path).is_file())
        html = Path(root_response.path).read_text(encoding="utf-8")
        self.assertEqual(Path(root_response.path), PROJECT_ROOT / "static" / "index.html")
        self.assertIn("<title>NovaShop Support</title>", html)
        self.assertIn("NOVASHOP AI SUPPORT", html)
        self.assertNotIn("GenAI Prompt Assistant", html)
        self.assertTrue((PROJECT_ROOT / "static" / "styles.css").is_file())
        script = PROJECT_ROOT / "static" / "app.js"
        self.assertTrue(script.is_file())
        self.assertIn('fetch("/api/chat"', script.read_text(encoding="utf-8"))
        self.assertEqual(health(), {"status": "ok"})
        api_routes = {(route.path, method) for route in app.routes if hasattr(route, "methods") for method in route.methods}
        self.assertIn(("/api/chat", "POST"), api_routes)


if __name__ == "__main__":
    unittest.main()
