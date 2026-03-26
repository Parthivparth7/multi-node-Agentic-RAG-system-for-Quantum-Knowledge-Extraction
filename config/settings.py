"""Central configuration for the Multi-Domain Knowledge Retrieval Agent."""

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
    quantum_pdfs_dir: Path = raw_pdfs_dir / "Quantum-Computing-Books"
    bio_pdfs_dir: Path = raw_pdfs_dir / "Bioinformatics-Books"
    processed_chunks_dir: Path = data_dir / "processed_chunks"
    chunks_file: Path = processed_chunks_dir / "chunks.jsonl"

    vector_db_dir: Path = project_root / "vector_db"
    faiss_quantum_general_dir: Path = vector_db_dir / "faiss_quantum_general"
    faiss_quantum_equations_dir: Path = vector_db_dir / "faiss_quantum_equations"
    faiss_quantum_hardware_dir: Path = vector_db_dir / "faiss_quantum_hardware"
    faiss_quantum_algorithms_dir: Path = vector_db_dir / "faiss_quantum_algorithms"

    faiss_bio_general_dir: Path = vector_db_dir / "faiss_bio_general"
    faiss_bio_terms_dir: Path = vector_db_dir / "faiss_bio_terms"
    faiss_bio_algorithms_dir: Path = vector_db_dir / "faiss_bio_algorithms"

    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model_name: str = os.getenv("QKRA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    fallback_embedding_dim: int = 384

    neo4j_uri: str = os.getenv("QKRA_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("QKRA_NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("QKRA_NEO4J_PASSWORD", "neo4j")


settings = Settings()

NODE_TO_DB_DIR = {
    "quantum_general": settings.faiss_quantum_general_dir,
    "quantum_equation": settings.faiss_quantum_equations_dir,
    "quantum_hardware": settings.faiss_quantum_hardware_dir,
    "quantum_algorithm": settings.faiss_quantum_algorithms_dir,
    "bio_general": settings.faiss_bio_general_dir,
    "bio_terms": settings.faiss_bio_terms_dir,
    "bio_algorithm": settings.faiss_bio_algorithms_dir,
}
