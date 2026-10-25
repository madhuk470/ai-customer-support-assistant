"""Live retrieval evaluation for the NovaShop policy corpus.

Run with the configured Gemini API available. This reports ranking and
coverage evidence; it does not interpret cosine distance as confidence.
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion import load_documents
from app.vector_store import index_documents, retrieve_documents
from app.config import DOCUMENTS_DIR
from app.gemini_client import embed_documents


EVALUATION_CASES = [
    {
        "name": "return window / direct",
        "question": "How many days do I have to return an order?",
        "evidence": "within 7 calendar days",
        "supported": True,
    },
    {
        "name": "return window / paraphrase",
        "question": "Can I send my purchase back a week after it arrives?",
        "evidence": "within 7 calendar days",
        "supported": True,
    },
    {
        "name": "delivery / direct",
        "question": "How long does standard delivery take?",
        "evidence": "3 to 7 business days after dispatch",
        "supported": True,
    },
    {
        "name": "delivery / paraphrase",
        "question": "When should a regular shipment arrive once it has been sent?",
        "evidence": "3 to 7 business days after dispatch",
        "supported": True,
    },
    {
        "name": "damaged item / direct",
        "question": "What should I do if I receive a damaged product?",
        "evidence": "contact NovaShop support within 48 hours",
        "supported": True,
    },
    {
        "name": "damaged item / paraphrase",
        "question": "A faulty item arrived. How do I report it?",
        "evidence": "contact NovaShop support within 48 hours",
        "supported": True,
    },
    {
        "name": "unsupported / cars",
        "question": "Does NovaShop sell cars?",
        "evidence": None,
        "supported": False,
    },
    {
        "name": "unsupported / store hours",
        "question": "What time does the NovaShop store close?",
        "evidence": None,
        "supported": False,
    },
]


def run_evaluation(top_k: int = 4, reindex: bool = False) -> list[dict]:
    """Run live cases and print evidence hit rank and top cosine distances."""
    if reindex:
        chunks = load_documents(DOCUMENTS_DIR)
        print("Reindexing section-aware chunks with Gemini document embeddings...")
        print(index_documents(chunks, embedding_fn=embed_documents))
    print("Metric: Chroma cosine distance (lower is closer), not confidence.")
    results = []
    for case in EVALUATION_CASES:
        retrieved = retrieve_documents(case["question"], top_k=top_k)
        evidence_rank = next(
            (rank for rank, item in enumerate(retrieved, start=1)
             if case["evidence"] and case["evidence"].casefold() in item["text"].casefold()),
            None,
        )
        results.append({**case, "results": retrieved, "evidence_rank": evidence_rank})
        print(f"\n{case['name']}: {case['question']}")
        coverage = retrieved[0]["coverage"] if retrieved else {"status": "no_results", "topics": []}
        print(f"  query coverage={coverage['status']} topics={coverage['topics']}")
        for rank, item in enumerate(retrieved, start=1):
            print(
                f"  #{rank} {item['id']} {item['source']} chunk={item['chunk_index']} "
                f"distance={item['distance']:.4f}"
            )
        if case["supported"]:
            print(f"  expected evidence rank={evidence_rank}")
        else:
            print("  corpus coverage: unsupported; returned neighbors are not evidence")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reindex", action="store_true", help="re-embed current documents before evaluation")
    parser.add_argument("--top-k", type=int, default=4)
    arguments = parser.parse_args()
    run_evaluation(top_k=arguments.top_k, reindex=arguments.reindex)
