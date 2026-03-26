"""Quantum hardware detection node."""

HARDWARE_KEYWORDS = {
    "superconducting": ["superconducting", "transmon", "josephson"],
    "trapped_ions": ["trapped ion", "ion trap"],
    "photonic": ["photonic", "linear optical", "optical quantum"],
}


def detect_hardware(chunk: str) -> dict[str, bool]:
    """Detect hardware families discussed in a text chunk."""
    text = chunk.lower()
    return {family: any(k in text for k in kws) for family, kws in HARDWARE_KEYWORDS.items()}
