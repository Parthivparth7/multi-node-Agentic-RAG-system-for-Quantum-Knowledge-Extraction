"""Equation extraction node."""

from __future__ import annotations

import re

_PATTERNS = [
    r"\$(.+?)\$",
    r"\\\[(.+?)\\\]",
    r"\\begin\{equation\}(.+?)\\end\{equation\}",
]


def extract_equations(text: str) -> list[str]:
    """Extract likely LaTeX/math expressions from text."""
    found: list[str] = []
    for pattern in _PATTERNS:
        found.extend(re.findall(pattern, text, flags=re.DOTALL))
    cleaned = [x.strip().replace("\n", " ") for x in found if x.strip()]
    return list(dict.fromkeys(cleaned))


def annotate_equations(records: list[dict]) -> list[dict]:
    """Keep only chunks with equations and attach extracted expression list."""
    out: list[dict] = []
    for rec in records:
        equations = extract_equations(rec["chunk"])
        if equations:
            enriched = dict(rec)
            enriched["equations"] = equations
            enriched["node"] = "equation"
            out.append(enriched)
    return out
