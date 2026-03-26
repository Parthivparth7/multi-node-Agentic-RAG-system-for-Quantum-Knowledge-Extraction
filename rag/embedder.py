"""Embedding utilities."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from config.settings import settings


class Embedder:
    """Thin wrapper around sentence-transformers."""

    def __init__(self, model_name: str = settings.embedding_model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of text items into dense vectors."""
        return self.model.encode(texts, normalize_embeddings=True).tolist()
