"""Embedding utilities with offline fallback."""

from __future__ import annotations

import hashlib
import logging
import math

from config.settings import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around sentence-transformers with offline-safe fallback."""

    def __init__(self, model_name: str = settings.embedding_model_name):
        self.model = None
        self.dim = settings.fallback_embedding_dim
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            logger.info("Loaded embedding model: %s", model_name)
        except Exception as exc:  # pragma: no cover - exercised in offline envs
            logger.warning("Falling back to hash embeddings due to model load failure: %s", exc)

    def _hash_embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of text items into dense vectors."""
        if self.model is not None:
            arr = self.model.encode(texts, normalize_embeddings=True)
            return arr.tolist()
        return [self._hash_embed(t) for t in texts]
