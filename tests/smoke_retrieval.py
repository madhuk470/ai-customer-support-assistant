"""Live Gemini + persistent Chroma retrieval smoke check (requires API access)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vector_store import retrieve_documents


QUESTIONS = [
    "How many days do I have to return an order?",
    "How long does standard delivery take?",
    "What should I do if I receive a damaged product?",
    "Does NovaShop sell cars?",
]


if __name__ == "__main__":
    print("Metric: Chroma cosine distance (lower is closer); it is not confidence and can rank irrelevant chunks highly.")
    for question in QUESTIONS:
        print(f"\nQUESTION: {question}")
        try:
            results = retrieve_documents(question, top_k=3)
        except Exception as error:
            print(f"  RETRIEVAL FAILED: {error}")
            continue
        for result in results:
            distance = result["distance"]
            distance_text = f"{distance:.4f}" if distance is not None else "unavailable"
            print(
                f"  id={result['id']} source={result['source']} chunk={result['chunk_index']} "
                f"distance={distance_text}\n  {result['text']}"
            )
