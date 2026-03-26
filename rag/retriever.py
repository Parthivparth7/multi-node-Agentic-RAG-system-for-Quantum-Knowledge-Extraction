"""FAISS retrieval helper."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissRetriever:
    """Load a FAISS index and aligned metadata to serve top-k retrieval."""

    def __init__(self, index_path: Path, metadata_path: Path):
        self.index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

    def search(self, query_vector: list[float], k: int = 5) -> list[dict]:
        """Return top-k metadata records with scores."""
        q = np.array([query_vector], dtype=np.float32)
        distances, ids = self.index.search(q, k)
        results = []
        for score, idx in zip(distances[0], ids[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
