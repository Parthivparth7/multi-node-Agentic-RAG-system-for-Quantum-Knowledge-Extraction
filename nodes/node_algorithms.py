"""Quantum algorithm detection node with hybrid semantic detection."""

from __future__ import annotations

import math
import re

from rag.embedder import Embedder

ALGORITHM_KEYWORDS = {
    "shor": ["shor", "factorization"],
    "grover": ["grover", "search speedup"],
    "vqe": ["vqe", "variational quantum eigensolver"],
    "qaoa": ["qaoa", "quantum approximate optimization algorithm"],
}

ALGORITHM_PROTOTYPES = {
    "shor": "Shor algorithm factorization quantum",
    "grover": "Grover search quantum speedup",
    "vqe": "Variational quantum eigensolver VQE Hamiltonian",
    "qaoa": "QAOA optimization quantum approximate",
}

_EMBEDDER = Embedder()
_PROTO_VECS = {k: _EMBEDDER.encode([v])[0] for k, v in ALGORITHM_PROTOTYPES.items()}


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def detect_algorithms(chunk: str, semantic_threshold: float = 0.45) -> dict[str, bool]:
    """Detect canonical algorithms via keyword + semantic similarity."""
    text = chunk.lower()
    chunk_vec = _EMBEDDER.encode([chunk])[0]
    out = {}
    for name, kws in ALGORITHM_KEYWORDS.items():
        keyword_hit = any(k in text for k in kws)
        semantic_hit = _cos(chunk_vec, _PROTO_VECS[name]) >= semantic_threshold
        out[name] = keyword_hit or semantic_hit
    return out


def extract_explanation_segment(chunk: str, keyword: str) -> str:
    """Extract sentence-level evidence for a specific algorithm mention."""
    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    keyword_lower = keyword.lower()
    for sentence in sentences:
        if keyword_lower in sentence.lower():
            return sentence.strip()
    return chunk[:220]


def annotate_algorithms(records: list[dict]) -> list[dict]:
    """Filter and annotate records for algorithm-specific retrieval."""
    out: list[dict] = []
    for rec in records:
        hits = detect_algorithms(rec["chunk"])
        active = [k for k, v in hits.items() if v]
        if active:
            evidence = {algo: extract_explanation_segment(rec["chunk"], algo) for algo in active}
            enriched = dict(rec)
            enriched["algorithms"] = active
            enriched["evidence"] = evidence
            enriched["node"] = "quantum_algorithm"
            out.append(enriched)
    return out
