"""Quantum hardware detection node."""

from __future__ import annotations

HARDWARE_KEYWORDS = {
    "superconducting": ["superconducting", "transmon", "josephson"],
    "trapped_ion": ["trapped ion", "ion trap"],
    "photonic": ["photonic", "linear optical", "optical quantum"],
}


def detect_hardware(chunk: str) -> dict[str, bool]:
    """Detect hardware families discussed in a text chunk."""
    text = chunk.lower()
    return {family: any(k in text for k in kws) for family, kws in HARDWARE_KEYWORDS.items()}


def annotate_hardware(records: list[dict]) -> list[dict]:
    """Filter/annotate records for hardware-specific index."""
    out: list[dict] = []
    for rec in records:
        hits = detect_hardware(rec["chunk"])
        if any(hits.values()):
            enriched = dict(rec)
            enriched["hardware_tags"] = [k for k, v in hits.items() if v]
            enriched["node"] = "hardware"
            out.append(enriched)
    return out
