import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vector_store import assess_query_coverage
from retrieval_eval import EVALUATION_CASES


class RetrievalEvaluationSetTests(unittest.TestCase):
    def test_evaluation_set_covers_direct_paraphrase_and_unsupported_queries(self):
        names = [case["name"] for case in EVALUATION_CASES]
        self.assertTrue(any("direct" in name for name in names))
        self.assertTrue(any("paraphrase" in name for name in names))
        self.assertTrue(any(not case["supported"] for case in EVALUATION_CASES))

    def test_unsupported_evaluation_queries_have_no_known_policy_topic(self):
        unsupported = [case for case in EVALUATION_CASES if not case["supported"]]
        self.assertGreaterEqual(len(unsupported), 2)
        for case in unsupported:
            with self.subTest(question=case["question"]):
                self.assertEqual(
                    assess_query_coverage(case["question"])["status"],
                    "no_known_topic",
                )

    def test_supported_direct_and_paraphrased_cases_map_to_policy_topics(self):
        supported = [case for case in EVALUATION_CASES if case["supported"]]
        self.assertTrue(all(assess_query_coverage(case["question"])["topics"] for case in supported))


if __name__ == "__main__":
    unittest.main()
