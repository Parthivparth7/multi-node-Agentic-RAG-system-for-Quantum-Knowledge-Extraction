"""FAISS retrieval helpers with pure-python fallback and reranking."""

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


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return _dot(a, b) / (na * nb)


class FaissStore:
    """Owns one FAISS index plus aligned metadata."""

    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.metadata: list[dict] = []
        self.vectors: list[list[float]] = []

    def build(self, embeddings, metadata: list[dict]) -> None:
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
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, str(_index_path(self.db_dir)))
        with _vectors_path(self.db_dir).open("w", encoding="utf-8") as f:
            json.dump(self.vectors, f)
        with _metadata_path(self.db_dir).open("w", encoding="utf-8") as f:
            for rec in self.metadata:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load(self) -> None:
        if faiss is not None and _index_path(self.db_dir).exists():
            self.index = faiss.read_index(str(_index_path(self.db_dir)))
        with _vectors_path(self.db_dir).open("r", encoding="utf-8") as f:
            self.vectors = json.load(f)
        with _metadata_path(self.db_dir).open("r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

    def search(self, query_vector: list[float], k: int = 3, candidate_k: int = 10, score_threshold: float = 0.3) -> list[dict]:
        """Search with threshold filtering and cosine reranking."""
        if not self.metadata:
            self.load()

        candidates: list[tuple[float, int]] = []
        if self.index is not None and faiss is not None and np is not None:
            q = np.array([query_vector], dtype=np.float32)
            scores, ids = self.index.search(q, candidate_k)
            candidates = [(float(s), int(i)) for s, i in zip(scores[0], ids[0]) if i >= 0]
        else:
            for i, v in enumerate(self.vectors):
                candidates.append((_cos(query_vector, v), i))
            candidates.sort(reverse=True)
            candidates = candidates[:candidate_k]

        reranked = []
        for _, idx in candidates:
            sim = _cos(query_vector, self.vectors[idx])
            if sim >= score_threshold:
                reranked.append((sim, idx))

        reranked.sort(reverse=True)
        results = []
        for sim, idx in reranked[:k]:
            row = dict(self.metadata[idx])
            row["score"] = float(sim)
            results.append(row)
        return results
