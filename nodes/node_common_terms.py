"""Common terms extraction node."""

from __future__ import annotations

from collections import Counter, defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None


def extract_common_terms(chunks: list[str], top_n: int = 100) -> list[str]:
    """Extract high-signal domain terms from corpus."""
    if not chunks:
        return []
    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(chunks)
        terms = vectorizer.get_feature_names_out()
        scores = matrix.mean(axis=0).A1
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:top_n]]

    counts = Counter()
    for c in chunks:
        tokens = [t.strip(".,:;!?()[]{}\"'").lower() for t in c.split()]
        counts.update(t for t in tokens if len(t) > 3)
    return [t for t, _ in counts.most_common(top_n)]


def annotate_chunks_with_terms(records: list[dict], top_n: int = 8) -> list[dict]:
    """Annotate each chunk with its local top terms for term-specific retrieval."""
    output = []
    for rec in records:
        terms = extract_common_terms([rec["chunk"]], top_n=top_n)
        enriched = dict(rec)
        enriched["terms"] = terms
        enriched["node"] = "common_terms"
        output.append(enriched)
    return output


def build_term_cooccurrence(records: list[dict]) -> dict[str, int]:
    """Build simple co-occurrence counts used by graph builder."""
    co = defaultdict(int)
    for rec in records:
        terms = rec.get("terms", [])[:5]
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pair = "||".join(sorted([terms[i], terms[j]]))
                co[pair] += 1
    return dict(co)
