"""PDF ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import fitz

from config.settings import settings
from rag.chunker import chunk_text


def load_pdf(path: Path) -> list[dict]:
    """Load a PDF and return page-level text records."""
    records: list[dict] = []
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            if text.strip():
                records.append({"source": path.name, "page": page.number + 1, "text": text})
    return records


def process_all(pdf_dir: Path = settings.raw_pdfs_dir) -> list[dict]:
    """Read PDFs and convert them into chunk records with metadata."""
    chunk_records: list[dict] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
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
    return chunk_records


def save_chunks(records: list[dict], output_file: Path | None = None) -> Path:
    """Persist chunk records as JSONL."""
    output_file = output_file or (settings.processed_chunks_dir / "chunks.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return output_file


if __name__ == "__main__":
    chunks = process_all()
    path = save_chunks(chunks)
    print(f"Saved {len(chunks)} chunks to {path}")
