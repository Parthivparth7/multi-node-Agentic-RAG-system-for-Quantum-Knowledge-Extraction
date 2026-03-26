"""Chunking utilities for PDF text."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


def chunk_text(text: str, chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap) -> list[str]:
    """Split text into overlapping chunks for retrieval.

    Args:
        text: Full document text.
        chunk_size: Maximum chunk length.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
