"""Embedding-based domain classifier for quantum vs bioinformatics."""

from __future__ import annotations

import math

from rag.embedder import Embedder


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class DomainClassifier:
    """Classify user query into quantum, bioinformatics, or cross-domain."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.prototypes = {
            "quantum": "qubit entanglement superposition quantum gate shor grover hamiltonian",
            "bioinformatics": "gene protein dna rna genome sequence blast alignment phylogeny",
        }
        proto_vectors = self.embedder.encode(list(self.prototypes.values()))
        self.proto_vec = {
            "quantum": proto_vectors[0],
            "bioinformatics": proto_vectors[1],
        }

    def classify(self, query: str, margin: float = 0.04) -> str:
        """Return quantum, bioinformatics, or cross if both are strong."""
        qv = self.embedder.encode([query])[0]
        q_score = _cos(qv, self.proto_vec["quantum"])
        b_score = _cos(qv, self.proto_vec["bioinformatics"])
        if abs(q_score - b_score) <= margin:
            return "cross"
        return "quantum" if q_score > b_score else "bioinformatics"
