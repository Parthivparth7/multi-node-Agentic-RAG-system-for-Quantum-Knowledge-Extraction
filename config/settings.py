"""Central configuration for the Quantum Knowledge Retrieval Agent (QKRA)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local/offline execution."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_pdfs_dir: Path = data_dir / "raw_pdfs"
    processed_chunks_dir: Path = data_dir / "processed_chunks"

    vector_db_dir: Path = project_root / "vector_db"
    faiss_common_terms_dir: Path = vector_db_dir / "faiss_common_terms"
    faiss_equations_dir: Path = vector_db_dir / "faiss_equations"
    faiss_hardware_dir: Path = vector_db_dir / "faiss_hardware"
    faiss_algorithms_dir: Path = vector_db_dir / "faiss_algorithms"
    faiss_general_dir: Path = vector_db_dir / "faiss_general"

    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()
