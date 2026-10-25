import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from app.ingestion import chunk_text, load_documents
except ModuleNotFoundError:
    # Allows this packaged test file to be checked before it is copied into
    # the project's tests folder beside the app package.
    from ingestion import chunk_text, load_documents


class ChunkTextTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text(" \n  "), [])

    def test_numbered_headings_stay_with_section_content(self):
        chunks = chunk_text(
            "Policy\n\n1. Return Window\nCustomers may return items within 7 days.\n\n"
            "2. Eligibility\nItems must be unused.",
            chunk_size=120,
        )
        combined = "\n".join(chunks)
        self.assertIn("1. Return Window\nCustomers may return items within 7 days.", combined)
        self.assertIn("2. Eligibility\nItems must be unused.", combined)
        self.assertNotIn("1.\n", combined)

    def test_long_paragraph_splits_at_sentence_boundary(self):
        text = "First policy sentence is clear. Second policy sentence stays intact. Third policy sentence follows."
        chunks = chunk_text(text, chunk_size=55, overlap=0)
        self.assertGreater(len(chunks), 1)
        self.assertIn("First policy sentence is clear.", chunks[0])
        self.assertTrue(all(not c.startswith("Second") or c.startswith("Second policy sentence") for c in chunks))

    def test_chunks_respect_character_limit_for_normal_tokens(self):
        text = " ".join(f"word{i}." for i in range(80))
        chunks = chunk_text(text, chunk_size=60)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 60 for chunk in chunks))

    def test_overlap_setting_does_not_duplicate_trailing_text(self):
        text = "Alpha sentence ends here. Beta sentence ends here. Gamma sentence ends here."
        chunks = chunk_text(text, chunk_size=32, overlap=12)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks).count("Alpha"), 1)
        self.assertEqual(" ".join(chunks).count("Beta"), 1)
        self.assertEqual(" ".join(chunks).count("Gamma"), 1)

    def test_source_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            documents_dir = Path(tmp)
            (documents_dir / "refund_policy.txt").write_text(
                "1. Return Window\nReturns are available within seven days.", encoding="utf-8"
            )
            documents = load_documents(documents_dir)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["id"], "refund_policy-0")
        self.assertIn("1. Return Window", documents[0]["text"])
        self.assertEqual(documents[0]["source"], "refund_policy.txt")
        self.assertEqual(documents[0]["chunk_index"], 0)

    def test_numbered_sections_are_separate_chunks_with_section_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            documents_dir = Path(tmp)
            (documents_dir / "refund_policy.txt").write_text(
                "NovaShop Returns\n\n1. Return Window\nReturns are available within seven days.\n\n"
                "2. Refunds\nApproved refunds are initiated within five days.",
                encoding="utf-8",
            )
            documents = load_documents(documents_dir)

        self.assertEqual(len(documents), 2)
        self.assertIn("Returns are available within seven days.", documents[0]["text"])
        self.assertEqual(documents[0]["section"], "1. Return Window")
        self.assertEqual(documents[1]["section"], "2. Refunds")


if __name__ == "__main__":
    unittest.main()
