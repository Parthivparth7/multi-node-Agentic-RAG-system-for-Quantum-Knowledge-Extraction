"""PDF ingestion pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import settings
from rag.chunker import chunk_text

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> list[dict]:
    """Load a PDF and return page-level text records."""
    records: list[dict] = []
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF ingestion") from exc

    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            if text.strip():
                records.append({"source": path.name, "page": page.number + 1, "text": text})
    return records


def process_all(pdf_dir: Path = settings.raw_pdfs_dir) -> list[dict]:
    """Read PDFs and convert them into chunk records with metadata."""
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    logger.info("Discovered %s PDF files in %s", len(pdf_paths), pdf_dir)
    chunk_records: list[dict] = []
    for pdf_path in pdf_paths:
        for rec in load_pdf(pdf_path):
            chunks = chunk_text(rec["text"])
            for idx, chunk in enumerate(chunks):
                chunk_records.append(
                    {
                        "id": f"{rec['source']}::p{rec['page']}::c{idx}",
                        "source": rec["source"],
                        "page": rec["page"],
                        "chunk": chunk,
                    }
                )
    logger.info("Created %s chunk records", len(chunk_records))
    return chunk_records


def save_chunks(records: list[dict], output_file: Path | None = None) -> Path:
    """Persist chunk records as JSONL."""
    output_file = output_file or settings.chunks_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved chunks to %s", output_file)
    return output_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = process_all()
    path = save_chunks(chunks)
    print(f"Saved {len(chunks)} chunks to {path}")
