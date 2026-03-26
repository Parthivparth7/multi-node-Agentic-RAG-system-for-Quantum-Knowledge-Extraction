"""Common terms extraction node."""

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_common_terms(chunks: list[str], top_n: int = 100) -> list[str]:
    """Extract high-signal terms from chunk corpus using TF-IDF."""
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max(top_n * 4, 400))
    matrix = vectorizer.fit_transform(chunks)
    scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [t for t, _ in pairs[:top_n]]
