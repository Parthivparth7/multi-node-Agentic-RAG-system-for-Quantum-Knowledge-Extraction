"""Quantum algorithm detection node."""

from __future__ import annotations

import re

ALGORITHM_KEYWORDS = {
    "shor": ["shor", "factorization"],
    "grover": ["grover", "search speedup"],
    "vqe": ["vqe", "variational quantum eigensolver"],
    "qaoa": ["qaoa", "quantum approximate optimization algorithm"],
}


def detect_algorithms(chunk: str) -> dict[str, bool]:
    """Detect if canonical quantum algorithms are mentioned."""
    text = chunk.lower()
    return {name: any(k in text for k in kws) for name, kws in ALGORITHM_KEYWORDS.items()}


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
            enriched["node"] = "algorithm"
            out.append(enriched)
    return out
