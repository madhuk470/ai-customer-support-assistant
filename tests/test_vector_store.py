import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from app.gemini_client import embed_documents, embed_query
from app.vector_store import assess_query_coverage, index_documents, retrieve_documents


class FakeEmbeddingResponse:
    def __init__(self, vectors):
        self.embeddings = [Mock(values=vector) for vector in vectors]


class GeminiEmbeddingTests(unittest.TestCase):
    def test_document_task_type_and_embedding_values(self):
        fake_client = Mock()
        fake_client.models.embed_content.return_value = FakeEmbeddingResponse([[0.1, 0.2]])

        vectors = embed_documents(["Policy text"], client_instance=fake_client)

        self.assertEqual(vectors, [[0.1, 0.2]])
        call = fake_client.models.embed_content.call_args.kwargs
        self.assertEqual(call["model"], "gemini-embedding-001")
        self.assertEqual(call["config"].task_type, "RETRIEVAL_DOCUMENT")

    def test_empty_documents_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty document list"):
            embed_documents([])

    def test_api_failure_has_actionable_error(self):
        fake_client = Mock()
        fake_client.models.embed_content.side_effect = OSError("network down")
        with self.assertRaisesRegex(RuntimeError, "connectivity"):
            embed_documents(["Policy text"], client_instance=fake_client)

    def test_query_uses_retrieval_query_task_type(self):
        fake_client = Mock()
        fake_client.models.embed_content.return_value = FakeEmbeddingResponse([[0.7, 0.8]])
        self.assertEqual(embed_query("  Return policy? ", client_instance=fake_client), [0.7, 0.8])
        call = fake_client.models.embed_content.call_args.kwargs
        self.assertEqual(call["model"], "gemini-embedding-001")
        self.assertEqual(call["contents"], "Return policy?")
        self.assertEqual(call["config"].task_type, "RETRIEVAL_QUERY")

    def test_empty_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty text"):
            embed_query("  ")


class PersistentIndexTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "chroma_db"
        self.documents = [
            {
                "id": "refund_policy-0",
                "text": "1. Return Window\nReturns within seven days.",
                "source": "refund_policy.txt",
                "chunk_index": 0,
            },
            {
                "id": "shipping_policy-0",
                "text": "1. Order Processing\nOrders process in two days.",
                "source": "shipping_policy.txt",
                "chunk_index": 0,
            },
        ]
        self.embedder = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    def tearDown(self):
        self.tempdir.cleanup()

    def test_index_persists_text_metadata_and_stable_ids(self):
        result = index_documents(
            self.documents,
            persist_directory=self.db_path,
            collection_name="test_policies",
            embedding_fn=self.embedder,
        )
        self.assertEqual(result["indexed_count"], 2)
        self.assertEqual(result["total_records"], 2)

        client = chromadb.PersistentClient(path=str(self.db_path))
        collection = client.get_collection("test_policies")
        stored = collection.get(ids=["refund_policy-0"], include=["documents", "metadatas"])
        self.assertEqual(stored["documents"], [self.documents[0]["text"]])
        self.assertEqual(
            stored["metadatas"],
            [{"source": "refund_policy.txt", "chunk_index": 0}],
        )
        client.close()

    def test_retrieval_returns_expected_fields_and_uses_embedding(self):
        index_documents(
            self.documents,
            persist_directory=self.db_path,
            collection_name="test_policies",
            embedding_fn=self.embedder,
        )
        query_embedder = Mock(return_value=[0.1, 0.2, 0.3])
        results = retrieve_documents(
            "How long can I return this?", top_k=1, persist_directory=self.db_path,
            collection_name="test_policies", embedding_fn=query_embedder,
        )
        query_embedder.assert_called_once_with("How long can I return this?")
        self.assertEqual(len(results), 1)
        self.assertEqual(
            set(results[0]), {"text", "source", "chunk_index", "id", "distance", "section", "section_match_topics", "coverage"}
        )
        self.assertEqual(results[0]["id"], "refund_policy-0")
        self.assertEqual(results[0]["source"], "refund_policy.txt")
        self.assertEqual(results[0]["chunk_index"], 0)
        self.assertIsNone(results[0]["section"])
        self.assertEqual(results[0]["coverage"]["status"], "recognized_policy_topic")
        self.assertIsInstance(results[0]["distance"], float)

    def test_query_coverage_recognizes_paraphrases_and_unsupported_topics(self):
        self.assertIn("returns", assess_query_coverage("Can I send my purchase back?")["topics"])
        self.assertIn("damaged_items", assess_query_coverage("A faulty item arrived")["topics"])
        self.assertEqual(assess_query_coverage("Does NovaShop sell cars?")["status"], "no_known_topic")
        self.assertEqual(assess_query_coverage("What time does the store close?")["topics"], [])

    def test_section_metadata_reranks_target_policy_section(self):
        section_documents = [
            {"id": "exceptions", "text": "6. Exceptions\nLate return requests may be rejected.",
             "source": "refund_policy.txt", "chunk_index": 5, "section": "6. Exceptions"},
            {"id": "return-window", "text": "1. Return Window\nReturns are available within 7 days.",
             "source": "refund_policy.txt", "chunk_index": 0, "section": "1. Return Window"},
        ]
        index_documents(
            section_documents,
            persist_directory=self.db_path,
            collection_name="section_rank_policies",
            embedding_fn=Mock(return_value=[[1.0, 0.0], [1.0, 0.0]]),
        )
        results = retrieve_documents(
            "Can I send my purchase back a week after it arrives?",
            top_k=1,
            persist_directory=self.db_path,
            collection_name="section_rank_policies",
            embedding_fn=Mock(return_value=[1.0, 0.0]),
        )
        self.assertEqual(results[0]["id"], "return-window")
        self.assertEqual(results[0]["section_match_topics"], ["returns"])

    def test_specific_damaged_item_intent_beats_incidental_arrival_topic(self):
        section_documents = [
            {"id": "shipment-damage", "text": "6. Lost or Damaged Shipments\nContact support about damaged packages.",
             "source": "shipping_policy.txt", "chunk_index": 5, "section": "6. Lost or Damaged Shipments"},
            {"id": "product-damage", "text": "4. Damaged or Incorrect Products\nReport a faulty product within 48 hours.",
             "source": "refund_policy.txt", "chunk_index": 3, "section": "4. Damaged or Incorrect Products"},
        ]
        index_documents(
            section_documents,
            persist_directory=self.db_path,
            collection_name="damage_rank_policies",
            embedding_fn=Mock(return_value=[[1.0, 0.0], [1.0, 0.0]]),
        )
        results = retrieve_documents(
            "A faulty item arrived. How do I report it?",
            top_k=1,
            persist_directory=self.db_path,
            collection_name="damage_rank_policies",
            embedding_fn=Mock(return_value=[1.0, 0.0]),
        )
        self.assertEqual(results[0]["id"], "product-damage")

    def test_retrieval_validates_question_and_top_k_before_embedding(self):
        query_embedder = Mock()
        for question, top_k in [(" ", 2), ("question", 0), ("question", -1), ("question", True), ("question", 1.5)]:
            with self.subTest(question=question, top_k=top_k):
                with self.assertRaises(ValueError):
                    retrieve_documents(question, top_k=top_k, embedding_fn=query_embedder)
        query_embedder.assert_not_called()

    def test_repeated_indexing_upserts_without_duplicates(self):
        first = index_documents(
            self.documents,
            persist_directory=self.db_path,
            collection_name="test_policies",
            embedding_fn=self.embedder,
        )
        second = index_documents(
            self.documents,
            persist_directory=self.db_path,
            collection_name="test_policies",
            embedding_fn=self.embedder,
        )
        self.assertEqual(first["total_records"], 2)
        self.assertEqual(second["total_records"], 2)

    def test_empty_input_is_rejected_before_embedding(self):
        with self.assertRaisesRegex(ValueError, "No document chunks"):
            index_documents([], persist_directory=self.db_path, embedding_fn=self.embedder)
        self.embedder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
