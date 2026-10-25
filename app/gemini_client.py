from google import genai
from google.genai import types

from app.config import EMBEDDING_MODEL, GEMINI_API_KEY


client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(prompt: str) -> str:
    """Generate an answer using the configured Gemini model."""
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )
    return response.text or ""


def embed_documents(texts: list[str], client_instance=None) -> list[list[float]]:
    """Embed policy chunks with Gemini's document retrieval task type."""
    if not texts:
        raise ValueError("Cannot create embeddings for an empty document list")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every document to embed must contain non-empty text")

    embedding_client = client_instance or client
    try:
        response = embedding_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings = response.embeddings
        if embeddings is None or len(embeddings) != len(texts):
            raise ValueError("Gemini returned a different number of embeddings than documents")
        vectors = [list(item.values or []) for item in embeddings]
        if any(not vector for vector in vectors):
            raise ValueError("Gemini returned an empty document embedding")
        return [[float(value) for value in vector] for vector in vectors]
    except ValueError:
        raise
    except Exception:
        raise RuntimeError(
            "Gemini document embedding failed. Check connectivity, API credentials, quota, and model availability."
        ) from None


def embed_query(question: str, client_instance=None) -> list[float]:
    """Embed one customer question for retrieval against policy documents."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must contain non-empty text")

    embedding_client = client_instance or client
    try:
        response = embedding_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question.strip(),
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        embeddings = response.embeddings
        if embeddings is None or len(embeddings) != 1:
            raise ValueError("Gemini returned an unexpected number of query embeddings")
        vector = list(embeddings[0].values or [])
        if not vector:
            raise ValueError("Gemini returned an empty query embedding")
        return [float(value) for value in vector]
    except ValueError:
        raise
    except Exception:
        raise RuntimeError(
            "Gemini query embedding failed. Check connectivity, API credentials, quota, and model availability."
        ) from None
