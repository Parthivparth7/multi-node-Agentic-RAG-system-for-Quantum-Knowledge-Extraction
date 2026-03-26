"""Quantum algorithm detection node."""

ALGORITHM_KEYWORDS = {
    "shor": ["shor"],
    "grover": ["grover"],
    "vqe": ["vqe", "variational quantum eigensolver"],
    "qaoa": ["qaoa", "quantum approximate optimization algorithm"],
}


def detect_algorithms(chunk: str) -> dict[str, bool]:
    """Detect if canonical quantum algorithms are mentioned."""
    text = chunk.lower()
    return {name: any(k in text for k in kws) for name, kws in ALGORITHM_KEYWORDS.items()}
