"""Bioinformatics node utilities with stricter entity extraction."""

from __future__ import annotations

import re

BIO_TERMS = {"gene", "protein", "dna", "rna", "genome", "transcriptome", "sequence", "motif", "proteome"}
BIO_ALGOS = {"blast", "smith-waterman", "needleman-wunsch", "alignment", "phylogeny"}
STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "into",
    "using",
    "helps",
    "analysis",
    "method",
    "model",
    "study",
}

GENE_PATTERN = re.compile(r"\b[A-Z0-9]{3,8}\b")
PROTEIN_PATTERN = re.compile(r"\b[A-Z][a-z]{2,12}(?:ase|in|ogen)\b")


def extract_bio_entities(chunk: str) -> list[str]:
    """Extract biologically meaningful entities and remove noise."""
    entities: list[str] = []
    lower_text = chunk.lower()

    for term in BIO_TERMS.union(BIO_ALGOS):
        if term in lower_text:
            entities.append(term)

    for m in GENE_PATTERN.findall(chunk):
        if m.lower() not in STOPWORDS:
            entities.append(m)

    for m in PROTEIN_PATTERN.findall(chunk):
        if m.lower() not in STOPWORDS:
            entities.append(m)

    clean = []
    for e in entities:
        if e.lower() not in STOPWORDS and len(e) >= 3:
            clean.append(e)
    return list(dict.fromkeys(clean))


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
