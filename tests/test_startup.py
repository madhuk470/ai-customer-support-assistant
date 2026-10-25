import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.startup import EXPECTED_EMBEDDING_DIMENSION, ensure_index
from app.vector_store import COLLECTION_NAME, index_documents


class StartupIndexTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.db_dir = self.root / "chroma"
        self.documents_dir = self.root / "documents"
        self.documents_dir.mkdir()
        (self.documents_dir / "refund_policy.txt").write_text(
            "1. Return Window\nCustomers can request a return within 7 calendar days.",
            encoding="utf-8",
        )
        self.embedding = Mock(
            side_effect=lambda texts: [[0.1] * EXPECTED_EMBEDDING_DIMENSION for _ in texts]
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_missing_collection_is_created_and_validated(self):
        result = ensure_index(self.db_dir, self.documents_dir, self.embedding)

        self.assertTrue(result["initialized"])
        self.assertEqual(result["indexed_count"], 1)
        self.assertEqual(result["total_records"], 1)
        self.embedding.assert_called_once()

    def test_empty_collection_is_initialized(self):
        client = chromadb.PersistentClient(path=str(self.db_dir))
        client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        client.close()

        result = ensure_index(self.db_dir, self.documents_dir, self.embedding)

        self.assertTrue(result["initialized"])
        self.assertEqual(result["total_records"], 1)
        self.embedding.assert_called_once()

    def test_valid_existing_collection_is_reused_without_embedding(self):
        source_documents = [{
            "id": "refund_policy-0",
            "text": "1. Return Window\nCustomers can request a return within 7 calendar days.",
            "source": "refund_policy.txt",
            "chunk_index": 0,
            "section": "1. Return Window",
        }]
        index_documents(
            source_documents,
            persist_directory=self.db_dir,
            embedding_fn=self.embedding,
        )
        self.embedding.reset_mock()

        result = ensure_index(self.db_dir, self.documents_dir, self.embedding)

        self.assertFalse(result["initialized"])
        self.assertEqual(result["total_records"], 1)
        self.embedding.assert_not_called()

    def test_invalid_populated_collection_fails_fast(self):
        client = chromadb.PersistentClient(path=str(self.db_dir))
        collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "l2"})
        collection.add(
            ids=["bad-index"],
            documents=["Not a valid cosine policy index"],
            metadatas=[{"source": "refund_policy.txt", "chunk_index": 0}],
            embeddings=[[0.1] * EXPECTED_EMBEDDING_DIMENSION],
        )
        client.close()

        with self.assertRaisesRegex(RuntimeError, "must use cosine distance"):
            ensure_index(self.db_dir, self.documents_dir, self.embedding)
        self.embedding.assert_not_called()

    def test_empty_documents_fail_startup_without_embedding(self):
        empty_dir = self.root / "empty-documents"
        empty_dir.mkdir()

        with self.assertRaisesRegex(RuntimeError, "No policy chunks found"):
            ensure_index(self.db_dir, empty_dir, self.embedding)
        self.embedding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
