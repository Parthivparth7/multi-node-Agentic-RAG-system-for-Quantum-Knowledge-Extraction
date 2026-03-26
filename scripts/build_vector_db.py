"""Build FAISS indexes for node-specific corpora."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from config.settings import settings
from rag.embedder import Embedder


def build_index(records: list[dict], output_dir: Path, embedder: Embedder) -> None:
    """Build and persist a single FAISS index + metadata file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [r["chunk"] for r in records]
    embeddings = np.array(embedder.encode(texts), dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[dict]:
    """Load chunk records from JSONL."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(chunks_file: Path = settings.processed_chunks_dir / "chunks.jsonl") -> None:
    """Build all FAISS databases (currently seeded from same corpus)."""
    records = load_chunks(chunks_file)
    embedder = Embedder()

    targets = {
        "common_terms": settings.faiss_common_terms_dir,
        "equations": settings.faiss_equations_dir,
        "hardware": settings.faiss_hardware_dir,
        "algorithms": settings.faiss_algorithms_dir,
        "general": settings.faiss_general_dir,
    }
    for _, out_dir in targets.items():
        build_index(records, out_dir, embedder)


if __name__ == "__main__":
    run()
    print("FAISS indexes built.")
