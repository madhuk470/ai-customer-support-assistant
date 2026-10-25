"""Initialize and validate the persistent policy index before serving traffic."""

from pathlib import Path
from typing import Callable

import chromadb
from chromadb.errors import NotFoundError

from app.config import CHROMA_DIR, DOCUMENTS_DIR
from app.ingestion import load_documents
from app.vector_store import COLLECTION_NAME, index_documents


EXPECTED_EMBEDDING_DIMENSION = 3072


def _validate_populated_collection(collection) -> int:
    """Fail fast if an existing collection cannot serve policy retrieval."""
    count = collection.count()
    if count <= 0:
        return 0

    metadata = collection.metadata or {}
    if metadata.get("hnsw:space") != "cosine":
        raise RuntimeError(
            f"Chroma collection {COLLECTION_NAME!r} must use cosine distance; "
            f"found {metadata.get('hnsw:space')!r}."
        )

    sample = collection.get(limit=1, include=["documents", "metadatas", "embeddings"])
    if not sample.get("ids") or not sample.get("documents") or not sample.get("metadatas"):
        raise RuntimeError(f"Chroma collection {COLLECTION_NAME!r} has incomplete records.")

    document = sample["documents"][0]
    record_metadata = sample["metadatas"][0] or {}
    if (
        not isinstance(document, str)
        or not document.strip()
        or not isinstance(record_metadata.get("source"), str)
        or not isinstance(record_metadata.get("chunk_index"), int)
    ):
        raise RuntimeError(f"Chroma collection {COLLECTION_NAME!r} has invalid policy metadata.")

    embeddings = sample.get("embeddings")
    if embeddings is None or len(embeddings) != 1 or len(embeddings[0]) != EXPECTED_EMBEDDING_DIMENSION:
        dimension = len(embeddings[0]) if embeddings is not None and len(embeddings) else None
        raise RuntimeError(
            f"Chroma collection {COLLECTION_NAME!r} must contain "
            f"{EXPECTED_EMBEDDING_DIMENSION}-dimension embeddings; found {dimension!r}."
        )
    return count


def ensure_index(
    chroma_dir: Path | str | None = None,
    documents_dir: Path | str | None = None,
    embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict:
    """Create a missing/empty collection, or reuse a valid populated one.

    Existing populated collections are checked without re-embedding. Invalid
    populated collections fail startup rather than being silently overwritten.
    """
    database_dir = Path(chroma_dir) if chroma_dir is not None else CHROMA_DIR
    source_dir = Path(documents_dir) if documents_dir is not None else DOCUMENTS_DIR
    database_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(database_dir))
    try:
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except NotFoundError:
            collection = None

        if collection is not None:
            count = collection.count()
            if count:
                _validate_populated_collection(collection)
                print(f"Knowledge base already exists: {count} chunks")
                return {
                    "initialized": False,
                    "total_records": count,
                    "collection_name": COLLECTION_NAME,
                    "persist_directory": str(database_dir),
                }

            # An empty collection can be safely recreated if its distance
            # configuration is wrong; there is no customer data to discard.
            if (collection.metadata or {}).get("hnsw:space") != "cosine":
                client.delete_collection(name=COLLECTION_NAME)
    finally:
        client.close()

    chunks = load_documents(source_dir)
    if not chunks:
        raise RuntimeError(f"No policy chunks found in {source_dir}; refusing to start without a knowledge base.")

    result = index_documents(
        chunks,
        persist_directory=database_dir,
        embedding_fn=embedding_fn,
    )

    verification_client = chromadb.PersistentClient(path=str(database_dir))
    try:
        collection = verification_client.get_collection(name=COLLECTION_NAME)
        verified_count = _validate_populated_collection(collection)
        if verified_count < len(chunks):
            raise RuntimeError(
                f"Chroma stored {verified_count} policy chunks, but {len(chunks)} were indexed."
            )
    finally:
        verification_client.close()

    print(f"Knowledge base initialized: {result}")
    return {"initialized": True, **result}


if __name__ == "__main__":
    ensure_index()
