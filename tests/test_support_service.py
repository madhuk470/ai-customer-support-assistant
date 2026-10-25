import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.support_service import INSUFFICIENT_CONTEXT_MESSAGE, answer_question


class GroundedSupportServiceTests(unittest.TestCase):
    def test_known_question_reuses_retrieval_and_grounded_answer_generation(self):
        retriever = Mock(return_value=[{
            "text": "Customers can request a return within 7 calendar days of receiving their order.",
            "source": "refund_policy.txt",
            "section": "1. Return Window",
            "chunk_index": 0,
            "id": "refund_policy-0",
            "distance": 0.26,
            "section_match_topics": ["returns"],
        }])
        answerer = Mock(return_value="You can request a return within 7 calendar days of delivery.")

        result = answer_question(
            "How many days do I have to return an order?",
            retriever=retriever,
            answerer=answerer,
        )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"][0]["section"], "1. Return Window")
        retriever.assert_called_once_with("How many days do I have to return an order?", top_k=4)
        prompt = answerer.call_args.args[0]
        self.assertIn("using only the policy excerpts", prompt)
        self.assertIn("refund_policy.txt", prompt)
        self.assertIn("within 7 calendar days", prompt)

    def test_unsupported_question_does_not_call_retrieval_or_generation(self):
        retriever = Mock()
        answerer = Mock()

        result = answer_question("Does NovaShop sell cars?", retriever, answerer)

        self.assertEqual(result["status"], "insufficient_context")
        self.assertEqual(result["answer"], INSUFFICIENT_CONTEXT_MESSAGE)
        self.assertEqual(result["sources"], [])
        retriever.assert_not_called()
        answerer.assert_not_called()

    def test_model_can_signal_insufficient_context(self):
        retriever = Mock(return_value=[{
            "text": "Returns are available within 7 days.",
            "source": "refund_policy.txt",
            "section": "1. Return Window",
            "chunk_index": 0,
            "id": "refund_policy-0",
            "distance": 0.26,
            "section_match_topics": ["returns"],
        }])
        result = answer_question(
            "How many days do I have to return an order?",
            retriever,
            Mock(return_value="INSUFFICIENT_CONTEXT"),
        )
        self.assertEqual(result["status"], "insufficient_context")
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
