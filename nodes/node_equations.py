"""Equation extraction node with validation."""

from __future__ import annotations

import re

_PATTERNS = [
    r"\$(.+?)\$",
    r"\\\[(.+?)\\\]",
    r"\\begin\{equation\}(.+?)\\end\{equation\}",
]


def _normalize(eq: str) -> str:
    return re.sub(r"\s+", " ", eq.strip().replace("\n", " "))


def _is_valid_equation(eq: str) -> bool:
    symbols = ["=", "+", "-", "*", "/", "^", "\\frac", "\\sum", "\\int"]
    return any(s in eq for s in symbols) and len(eq) >= 3


def extract_equations(text: str) -> list[str]:
    """Extract likely LaTeX/math expressions from text."""
    found: list[str] = []
    for pattern in _PATTERNS:
        found.extend(re.findall(pattern, text, flags=re.DOTALL))
    normalized = [_normalize(x) for x in found if x.strip()]
    filtered = [e for e in normalized if _is_valid_equation(e)]
    return list(dict.fromkeys(filtered))


def annotate_equations(records: list[dict]) -> list[dict]:
    """Keep only chunks with valid equations and attach expression list."""
    out: list[dict] = []
    for rec in records:
        equations = extract_equations(rec["chunk"])
        if equations:
            enriched = dict(rec)
            enriched["equations"] = equations
            enriched["node"] = "quantum_equation"
            out.append(enriched)
    return out
