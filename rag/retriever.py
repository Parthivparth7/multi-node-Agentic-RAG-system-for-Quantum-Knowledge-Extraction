"""FAISS retrieval helpers with pure-python fallback."""

from __future__ import annotations

import json
import math
from pathlib import Path

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _index_path(db_dir: Path) -> Path:
    return db_dir / "index.faiss"


def _metadata_path(db_dir: Path) -> Path:
    return db_dir / "metadata.jsonl"


def _vectors_path(db_dir: Path) -> Path:
    return db_dir / "vectors.json"


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class FaissStore:
    """Owns one FAISS index plus aligned metadata."""

    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.metadata: list[dict] = []
        self.vectors: list[list[float]] = []

    def build(self, embeddings, metadata: list[dict]) -> None:
        """Build the index from vectors + metadata."""
        if np is not None and isinstance(embeddings, np.ndarray):
            vectors = embeddings.astype("float32").tolist()
            dim = embeddings.shape[1]
        else:
            vectors = [list(map(float, v)) for v in embeddings]
            dim = len(vectors[0]) if vectors else 0

        if faiss is not None and np is not None and vectors:
            index = faiss.IndexFlatIP(dim)
            index.add(np.array(vectors, dtype=np.float32))
            self.index = index
        self.vectors = vectors
        self.metadata = metadata

    def save(self) -> None:
        """Persist index and metadata."""
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, str(_index_path(self.db_dir)))
        with _vectors_path(self.db_dir).open("w", encoding="utf-8") as f:
            json.dump(self.vectors, f)
        with _metadata_path(self.db_dir).open("w", encoding="utf-8") as f:
            for rec in self.metadata:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load(self) -> None:
        """Load index and metadata from disk."""
        if faiss is not None and _index_path(self.db_dir).exists():
            self.index = faiss.read_index(str(_index_path(self.db_dir)))
        with _vectors_path(self.db_dir).open("r", encoding="utf-8") as f:
            self.vectors = json.load(f)
        with _metadata_path(self.db_dir).open("r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

    def search(self, query_vector: list[float], k: int = 5) -> list[dict]:
        """Return top-k records with similarity scores."""
        if not self.metadata:
            self.load()

        if self.index is not None and faiss is not None and np is not None:
            q = np.array([query_vector], dtype=np.float32)
            scores, ids = self.index.search(q, k)
            rows = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                row = dict(self.metadata[idx])
                row["score"] = float(score)
                rows.append(row)
            return rows

        qn = math.sqrt(sum(x * x for x in query_vector)) or 1.0
        scored = []
        for i, v in enumerate(self.vectors):
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            score = _dot(query_vector, v) / (qn * vn)
            scored.append((score, i))
        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:k]:
            row = dict(self.metadata[idx])
            row["score"] = float(score)
            results.append(row)
        return results
