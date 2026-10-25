from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Prefer the project-local .env copied from .env.example. Keep the historical
# parent-level .env as a fallback for existing local installations.
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
PARENT_ENV_FILE = PROJECT_ROOT.parent / ".env"
load_dotenv(PROJECT_ENV_FILE, override=False)
load_dotenv(PARENT_ENV_FILE, override=False)

_chroma_dir = os.getenv("CHROMA_DIR", "").strip()
CHROMA_DIR = Path(_chroma_dir).expanduser() if _chroma_dir else DATA_DIR / "chroma_db"
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = PROJECT_ROOT / CHROMA_DIR

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.5-flash-lite"

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. Set it in the process environment or in "
        f"{PROJECT_ENV_FILE}."
    )
