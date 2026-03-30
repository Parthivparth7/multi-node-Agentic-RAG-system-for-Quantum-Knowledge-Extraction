"""Build independent domain-specific FAISS indexes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import NODE_TO_DB_DIR, settings
from nodes.node_algorithms import annotate_algorithms
from nodes.node_bio import annotate_bio_algorithms, annotate_bio_terms
from nodes.node_equations import annotate_equations
from nodes.node_hardware import annotate_hardware
from rag.embedder import Embedder
from rag.retriever import FaissStore

logger = logging.getLogger(__name__)


def load_chunks(path: Path = settings.chunks_file) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _build_single(records: list[dict], db_dir: Path, embedder: Embedder) -> int:
    if not records:
        db_dir.mkdir(parents=True, exist_ok=True)
        return 0
    vectors = embedder.encode([r["chunk"] for r in records])
    store = FaissStore(db_dir)
    store.build(vectors, records)
    store.save()
    return len(records)


def run(chunks_file: Path = settings.chunks_file) -> dict[str, int]:
    """Build all domain-specific indexes and avoid domain mixing."""
    all_records = load_chunks(chunks_file)
    quantum = [r for r in all_records if r.get("domain") == "quantum"]
    bio = [r for r in all_records if r.get("domain") == "bioinformatics"]

    embedder = Embedder()
    node_records = {
        "quantum_general": [dict(r, node="quantum_general") for r in quantum],
        "quantum_equation": annotate_equations(quantum),
        "quantum_hardware": annotate_hardware(quantum),
        "quantum_algorithm": annotate_algorithms(quantum),
        "bio_general": [dict(r, node="bio_general") for r in bio],
        "bio_terms": annotate_bio_terms(bio),
        "bio_algorithm": annotate_bio_algorithms(bio),
    }

    counts = {}
    for node, records in node_records.items():
        counts[node] = _build_single(records, NODE_TO_DB_DIR[node], embedder)
        logger.info("Built %s with %s records", node, counts[node])
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run())
