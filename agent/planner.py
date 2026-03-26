"""Planner for selecting retrieval strategy."""

from __future__ import annotations


def route_query(query: str) -> str:
    """Route query to best-fit specialized node."""
    q = query.lower()
    if any(x in q for x in ["equation", "latex", "hamiltonian", "formula"]):
        return "equation"
    if any(x in q for x in ["hardware", "superconducting", "ion trap", "photonic"]):
        return "hardware"
    if any(x in q for x in ["algorithm", "shor", "grover", "vqe", "qaoa"]):
        return "algorithm"
    if any(x in q for x in ["term", "keyword", "glossary"]):
        return "common_terms"
    return "general"
