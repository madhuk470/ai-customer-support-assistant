from pathlib import Path
import re
from typing import Callable

import chromadb

from app.config import CHROMA_DIR
from app.gemini_client import embed_documents, embed_query


COLLECTION_NAME = "novashop_policies"

_COVERAGE_PATTERNS = {
    "returns": r"\b(return|returns|returning|send(?:ing)?\s+(?:(?:my|the|this|that)\s+)?(?:it|item|purchase|order)?\s*back|take\s+(?:it|item)\s+back)\b",
    "late_returns": r"\b(after|past|beyond)\s+(?:the\s+)?(?:7[- ]day|seven[- ]day|return window)|\b(late|overdue)\s+returns?\b",
    "eligibility": r"\b(eligible|eligibility|unused|opened|original packaging)\b",
    "refunds": r"\b(refund|refunds|reimburse|money back)\b",
    "delivery": r"\b(deliver|delivery|shipping|shipment|ship|arriv(?:e|es|ed|al)|dispatch|courier)\b",
    "damaged_items": r"\b(damage|damaged|broken|faulty|defective|incorrect|wrong (?:item|product))\b",
    "tracking": r"\b(track|tracking)\b",
    "delays": r"\b(late|delay|delayed|not arrived|missing shipment)\b",
    "addresses": r"\b(address|address change|delivery address)\b",
}

_TOPIC_SECTIONS = {
    "returns": {"1. return window"},
    "late_returns": {"6. exceptions"},
    "eligibility": {"2. eligibility"},
    "refunds": {"3. refund processing", "5. refund status"},
    "delivery": {"1. order processing", "2. standard delivery", "4. delayed deliveries", "6. lost or damaged shipments"},
    "damaged_items": {"4. damaged or incorrect products", "6. lost or damaged shipments"},
    "tracking": {"3. tracking orders"},
    "delays": {"4. delayed deliveries", "6. lost or damaged shipments"},
    "addresses": {"5. incorrect delivery address"},
}


def assess_query_coverage(question: str) -> dict:
    """Identify recognized NovaShop policy topics without using vector distance.

    This small topic map detects whether the question names a topic represented
    by the current policy collection. It is not a guarantee that the exact
    question is answered; unknown wording and new policy areas need evaluation.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must contain non-empty text")
    normalized = question.casefold()
    topics = [
        topic for topic, pattern in _COVERAGE_PATTERNS.items()
        if re.search(pattern, normalized)
    ]
    return {
        "status": "recognized_policy_topic" if topics else "no_known_topic",
        "topics": topics,
    }


def _section_match_topics(
    question: str, topics: list[str], section: str | None
) -> list[str]:
    # Generic arrival wording often appears as context in return or damage
    # questions. Prefer the more specific policy intent unless the customer
    # explicitly asks about shipping or delivery.
    explicit_delivery = re.search(
        r"\b(delivery|shipping|shipment|ship|dispatch|courier)\b",
        question.casefold(),
    )
    if not explicit_delivery and any(
        topic in topics for topic in ("returns", "late_returns", "damaged_items", "eligibility", "refunds")
    ):
        topics = [topic for topic in topics if topic != "delivery"]
    normalized_section = (section or "").casefold()
    if "damaged_items" in topics and re.search(r"\b(item|product|goods)\b", question.casefold()):
        if normalized_section == "6. lost or damaged shipments":
            return []
    if "damaged_items" in topics and re.search(r"\b(package|parcel|shipment)\b", question.casefold()):
        if normalized_section == "4. damaged or incorrect products":
            return []
    return [
        topic for topic in topics
        if normalized_section in _TOPIC_SECTIONS.get(topic, set())
    ]


def retrieve_documents(
    question: str,
    top_k: int = 3,
    persist_directory: Path | str | None = None,
    collection_name: str = COLLECTION_NAME,
    embedding_fn: Callable[[str], list[float]] | None = None,
) -> list[dict]:
    """Retrieve policy chunks ranked by Chroma cosine distance.

    Chroma's cosine distance is lower for closer vectors. It is a retrieval
    ranking signal, not a probability or a guarantee that a chunk answers the
    question.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must contain non-empty text")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    question = question.strip()
    coverage = assess_query_coverage(question)
    query_vector = (embedding_fn or embed_query)(question)
    db_path = Path(persist_directory) if persist_directory else CHROMA_DIR
    chroma_client = None
    try:
        chroma_client = chromadb.PersistentClient(path=str(db_path))
        collection = chroma_client.get_collection(name=collection_name)
        candidate_k = min(collection.count(), max(top_k, top_k * 4))
        if candidate_k == 0:
            return []
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        raise RuntimeError(
            f"ChromaDB could not retrieve policy chunks from {db_path}. Check that the collection exists and is readable."
        ) from None
    finally:
        if chroma_client is not None:
            chroma_client.close()

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    retrieved = []
    for index, identifier in enumerate(ids):
        metadata = metadatas[index] or {}
        retrieved.append({
            "text": documents[index],
            "source": metadata.get("source"),
            "chunk_index": metadata.get("chunk_index"),
            "id": identifier,
            "distance": distances[index] if index < len(distances) else None,
            "section": metadata.get("section"),
            "section_match_topics": _section_match_topics(
                question, coverage["topics"], metadata.get("section")
            ),
            "coverage": coverage,
        })
    # Use explicit section metadata as a reranking feature, while retaining the
    # original cosine distance for inspection. Unknown topics remain distance-
    # ranked and are not treated as covered based on a close neighbor.
    retrieved.sort(
        key=lambda item: (
            -len(item["section_match_topics"]),
            item["distance"] if item["distance"] is not None else float("inf"),
        )
    )
    return retrieved[:top_k]


def index_documents(
    documents: list[dict],
    persist_directory: Path | str | None = None,
    collection_name: str = COLLECTION_NAME,
    embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict:
    """Embed and upsert ingested chunks into a persistent Chroma collection."""
    if not documents:
        raise ValueError("No document chunks were provided for indexing")

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for position, item in enumerate(documents):
        if not isinstance(item, dict):
            raise ValueError(f"Document at position {position} must be a mapping")
        identifier = item.get("id")
        text = item.get("text")
        source = item.get("source")
        chunk_index = item.get("chunk_index")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Document at position {position} is missing a stable id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Document {identifier!r} has empty text")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Document {identifier!r} is missing its source filename")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
            raise ValueError(f"Document {identifier!r} has an invalid chunk_index")
        ids.append(identifier)
        texts.append(text)
        metadata = {"source": source, "chunk_index": chunk_index}
        section = item.get("section")
        if isinstance(section, str) and section.strip():
            metadata["section"] = section.strip()
        metadatas.append(metadata)

    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique within one indexing batch")

    make_embeddings = embedding_fn or embed_documents
    vectors = make_embeddings(texts)
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding provider returned a different number of vectors than chunks")
    if any(not vector for vector in vectors):
        raise RuntimeError("Embedding provider returned an empty vector")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise RuntimeError("Embedding provider returned vectors with inconsistent dimensions")

    db_path = Path(persist_directory) if persist_directory else CHROMA_DIR
    chroma_client = None
    try:
        chroma_client = chromadb.PersistentClient(path=str(db_path))
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )
        total_records = collection.count()
    except Exception:
        raise RuntimeError(
            f"ChromaDB could not persist document chunks in {db_path}. Check the directory and storage permissions."
        ) from None
    finally:
        if chroma_client is not None:
            chroma_client.close()

    return {
        "indexed_count": len(ids),
        "total_records": total_records,
        "collection_name": collection_name,
        "persist_directory": str(db_path),
    }
