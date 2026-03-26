"""Central configuration for the Quantum Knowledge Retrieval Agent (QKRA)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local/offline execution."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_pdfs_dir: Path = data_dir / "raw_pdfs"
    processed_chunks_dir: Path = data_dir / "processed_chunks"
    chunks_file: Path = processed_chunks_dir / "chunks.jsonl"

    vector_db_dir: Path = project_root / "vector_db"
    faiss_common_terms_dir: Path = vector_db_dir / "faiss_common_terms"
    faiss_equations_dir: Path = vector_db_dir / "faiss_equations"
    faiss_hardware_dir: Path = vector_db_dir / "faiss_hardware"
    faiss_algorithms_dir: Path = vector_db_dir / "faiss_algorithms"
    faiss_general_dir: Path = vector_db_dir / "faiss_general"

    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model_name: str = os.getenv("QKRA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    fallback_embedding_dim: int = 384

    neo4j_uri: str = os.getenv("QKRA_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("QKRA_NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("QKRA_NEO4J_PASSWORD", "neo4j")


settings = Settings()

NODE_TO_DB_DIR = {
    "common_terms": settings.faiss_common_terms_dir,
    "equation": settings.faiss_equations_dir,
    "hardware": settings.faiss_hardware_dir,
    "algorithm": settings.faiss_algorithms_dir,
    "general": settings.faiss_general_dir,
}
