"""Build independent FAISS indexes for each node pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import NODE_TO_DB_DIR, settings
from nodes.node_algorithms import annotate_algorithms
from nodes.node_common_terms import annotate_chunks_with_terms
from nodes.node_equations import annotate_equations
from nodes.node_hardware import annotate_hardware
from rag.embedder import Embedder
from rag.retriever import FaissStore

logger = logging.getLogger(__name__)


def load_chunks(path: Path = settings.chunks_file) -> list[dict]:
    """Load chunk records from JSONL."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _build_single(records: list[dict], db_dir: Path, embedder: Embedder) -> int:
    if not records:
        db_dir.mkdir(parents=True, exist_ok=True)
        return 0
    texts = [r["chunk"] for r in records]
    vectors = embedder.encode(texts)
    store = FaissStore(db_dir)
    store.build(vectors, records)
    store.save()
    return len(records)


def run(chunks_file: Path = settings.chunks_file) -> dict[str, int]:
    """Build all node-specific FAISS databases independently."""
    all_records = load_chunks(chunks_file)
    embedder = Embedder()

    node_records = {
        "common_terms": annotate_chunks_with_terms(all_records),
        "equation": annotate_equations(all_records),
        "hardware": annotate_hardware(all_records),
        "algorithm": annotate_algorithms(all_records),
        "general": [dict(r, node="general") for r in all_records],
    }

    counts: dict[str, int] = {}
    for node, records in node_records.items():
        counts[node] = _build_single(records, NODE_TO_DB_DIR[node], embedder)
        logger.info("Built %s index with %s records", node, counts[node])
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = run()
    print(output)
