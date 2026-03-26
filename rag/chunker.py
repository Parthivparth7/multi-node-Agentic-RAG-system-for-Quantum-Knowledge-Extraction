"""Chunking utilities for PDF text."""

from __future__ import annotations

from config.settings import settings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None


def chunk_text(text: str, chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap) -> list[str]:
    """Split text into overlapping chunks for retrieval."""
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    # fallback splitter
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks
