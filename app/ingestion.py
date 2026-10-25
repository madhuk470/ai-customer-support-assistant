import re
from pathlib import Path


SECTION_HEADING_RE = re.compile(r"^\s*\d+\.\s+\S.*$")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _section_units(text: str, chunk_size: int) -> list[str]:
    """Keep each numbered policy section intact and split oversized sections."""
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    found_heading = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current and current[-1] != "":
                current.append("")
            continue
        if SECTION_HEADING_RE.match(stripped):
            if current and found_heading:
                blocks.append("\n".join(current).strip())
                current = [stripped]
            elif current:
                # Drop any document-level preamble so the heading stays next
                # to the section text and can be embedded as a clear label.
                current = [stripped]
            else:
                current = [stripped]
            found_heading = True
        else:
            current.append(stripped)

    if current:
        blocks.append("\n".join(current).strip())

    units: list[str] = []
    for block in blocks:
        if len(block) <= chunk_size:
            units.append(block)
            continue

        lines_in_block = block.splitlines()
        heading = lines_in_block[0] if SECTION_HEADING_RE.match(lines_in_block[0]) else ""
        body = "\n".join(lines_in_block[1:]).strip() if heading else block
        sentences = [s.strip() for s in SENTENCE_BOUNDARY_RE.split(body) if s.strip()]
        part = heading
        for sentence in sentences:
            separator = "\n" if part else ""
            candidate = f"{part}{separator}{sentence}".strip()
            if len(candidate) <= chunk_size:
                part = candidate
                continue
            if part:
                units.append(part)
            part = sentence

        if part:
            units.append(part)

    return units


def chunk_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[str]:
    """Build bounded chunks, keeping numbered policy sections together when possible.

    The overlap argument remains accepted for compatibility. No overlap is
    emitted, because duplicated trailing text made adjacent policy chunks
    harder to read and retrieve accurately.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    text = clean_text(text)
    if not text:
        return []

    units = _section_units(text, chunk_size)
    return units


def load_documents(documents_dir: Path) -> list[dict]:
    """Load TXT files and retain stable source and chunk metadata."""
    documents: list[dict] = []
    for file_path in sorted(documents_dir.glob("*.txt")):
        text = clean_text(file_path.read_text(encoding="utf-8"))
        for index, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f"{file_path.stem}-{index}",
                    "text": chunk,
                    "source": file_path.name,
                    "chunk_index": index,
                    "section": chunk.splitlines()[0]
                    if chunk.splitlines() and SECTION_HEADING_RE.match(chunk.splitlines()[0])
                    else None,
                }
            )
    return documents


if __name__ == "__main__":
    from app.config import DOCUMENTS_DIR

    chunks = load_documents(DOCUMENTS_DIR)
    print(f"Documents directory: {DOCUMENTS_DIR}")
    print(f"Total chunks created: {len(chunks)}")
    for item in chunks:
        print(f"\nSource: {item['source']} | Chunk: {item['chunk_index']}")
        print(item["text"])
