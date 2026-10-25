"""FastAPI application for NovaShop's customer-support dashboard."""

from pathlib import Path
import logging
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import FileResponse

from app.support_service import answer_question


logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)


class SourceResponse(BaseModel):
    source: str | None = None
    section: str | None = None
    chunk_index: int | None = None
    id: str | None = None
    distance: float | None = None


class CoverageResponse(BaseModel):
    status: str
    topics: list[str]


class AnswerResponse(BaseModel):
    answer: str
    status: Literal["answered", "insufficient_context"]
    sources: list[SourceResponse]
    coverage: CoverageResponse


def get_answer_service():
    """Dependency seam for endpoint tests and alternative service wiring."""
    return answer_question


app = FastAPI(title="NovaShop Support", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=AnswerResponse)
def submit_question(
    payload: QuestionRequest,
    service=Depends(get_answer_service),
):
    try:
        return service(payload.question)
    except Exception:
        logger.exception("NovaShop support request failed")
        raise HTTPException(
            status_code=503,
            detail="The support service is temporarily unavailable. Please try again.",
        ) from None
