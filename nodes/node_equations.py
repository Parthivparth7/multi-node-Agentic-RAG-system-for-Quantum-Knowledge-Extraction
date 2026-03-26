"""Equation extraction node."""

import re

_LATEX_PATTERNS = [
    r"\$(.+?)\$",
    r"\\\[(.+?)\\\]",
    r"\\begin\{equation\}(.+?)\\end\{equation\}",
]


def extract_equations(text: str) -> list[str]:
    """Extract likely equation expressions from text."""
    eqs: list[str] = []
    for pattern in _LATEX_PATTERNS:
        eqs.extend(re.findall(pattern, text, flags=re.DOTALL))
    return [e.strip() for e in eqs if e.strip()]
