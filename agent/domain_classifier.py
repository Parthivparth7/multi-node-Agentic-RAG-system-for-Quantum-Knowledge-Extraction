"""Embedding-based domain classifier for quantum vs bioinformatics."""

from __future__ import annotations

import math

from rag.embedder import Embedder


def _norm(v: list[float]) -> list[float]:
    mag = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / mag for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    an = _norm(a)
    bn = _norm(b)
    return sum(x * y for x, y in zip(an, bn))


class DomainClassifier:
    """Classify user query into quantum, bioinformatics, or cross-domain."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.prototypes = {
            "quantum": "qubit entanglement superposition quantum gate shor grover hamiltonian",
            "bioinformatics": "gene protein dna rna genome sequence blast alignment phylogeny",
        }
        proto_vectors = self.embedder.encode(list(self.prototypes.values()))
        self.proto_vec = {"quantum": _norm(proto_vectors[0]), "bioinformatics": _norm(proto_vectors[1])}

    def classify_with_confidence(self, query: str, margin: float = 0.04) -> dict:
        """Return domain and confidence for robust routing."""
        qv = self.embedder.encode([query])[0]
        q_score = _cos(qv, self.proto_vec["quantum"])
        b_score = _cos(qv, self.proto_vec["bioinformatics"])
        if abs(q_score - b_score) <= margin:
            return {"domain": "cross", "confidence": round(1.0 - abs(q_score - b_score), 3), "scores": {"quantum": q_score, "bioinformatics": b_score}}
        if q_score > b_score:
            return {"domain": "quantum", "confidence": round(q_score, 3), "scores": {"quantum": q_score, "bioinformatics": b_score}}
        return {"domain": "bioinformatics", "confidence": round(b_score, 3), "scores": {"quantum": q_score, "bioinformatics": b_score}}

    def classify(self, query: str, margin: float = 0.04) -> str:
        """Backward-compatible domain output string."""
        return self.classify_with_confidence(query, margin=margin)["domain"]
