"""Planner for selecting domain-specific retrieval strategy."""

from __future__ import annotations


def route_query(query: str, domain: str) -> str:
    """Route query to best-fit specialized node in a domain."""
    q = query.lower()
    if domain == "quantum":
        if any(x in q for x in ["equation", "latex", "hamiltonian", "formula"]):
            return "quantum_equation"
        if any(x in q for x in ["hardware", "superconducting", "ion trap", "photonic"]):
            return "quantum_hardware"
        if any(x in q for x in ["algorithm", "shor", "grover", "vqe", "qaoa"]):
            return "quantum_algorithm"
        return "quantum_general"

    if any(x in q for x in ["blast", "alignment", "needleman", "smith-waterman"]):
        return "bio_algorithm"
    if any(x in q for x in ["gene", "protein", "dna", "rna", "genome", "sequence"]):
        return "bio_terms"
    return "bio_general"
