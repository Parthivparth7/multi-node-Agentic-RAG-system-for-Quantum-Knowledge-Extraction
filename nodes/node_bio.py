"""Bioinformatics node utilities."""

from __future__ import annotations

import re

BIO_TERMS = ["gene", "protein", "dna", "rna", "genome", "transcriptome", "sequence", "motif"]
BIO_ALGOS = ["blast", "smith-waterman", "needleman-wunsch", "alignment", "phylogeny"]


def extract_bio_entities(chunk: str) -> list[str]:
    """Extract gene/protein style entities and key bio terms."""
    words = re.findall(r"\b[A-Za-z0-9\-]{3,}\b", chunk)
    entities = []
    for w in words:
        lw = w.lower()
        if lw in BIO_TERMS or lw in BIO_ALGOS:
            entities.append(lw)
        elif re.match(r"^[A-Z0-9]{3,8}$", w):
            entities.append(w)
    return list(dict.fromkeys(entities))


def annotate_bio_terms(records: list[dict]) -> list[dict]:
    """Annotate chunks containing gene/protein/sequence terminology."""
    out = []
    for rec in records:
        entities = extract_bio_entities(rec["chunk"])
        if entities:
            e = dict(rec)
            e["bio_entities"] = entities
            e["node"] = "bio_terms"
            out.append(e)
    return out


def annotate_bio_algorithms(records: list[dict]) -> list[dict]:
    """Annotate chunks containing bioinformatics algorithm mentions."""
    out = []
    for rec in records:
        text = rec["chunk"].lower()
        hits = [a for a in BIO_ALGOS if a in text]
        if hits:
            e = dict(rec)
            e["bio_algorithms"] = hits
            e["node"] = "bio_algorithm"
            out.append(e)
    return out
