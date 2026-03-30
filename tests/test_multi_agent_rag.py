from pathlib import Path

from scripts.multi_agent_rag import MultiAgentRAGOrchestrator


def _write_store(path: Path, vectors: list[list[float]], chunks: list[str], domain: str):
    path.mkdir(parents=True, exist_ok=True)
    (path / "vectors.json").write_text(__import__("json").dumps(vectors), encoding="utf-8")
    with (path / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(__import__("json").dumps({"chunk": c, "domain": domain, "source": f"{domain}.pdf", "page": i + 1}) + "\n")


def test_multi_agent_pipeline(tmp_path: Path):
    dbs = {
        "quantum_general": tmp_path / "q",
        "bio_general": tmp_path / "b",
    }
    _write_store(dbs["quantum_general"], [[1.0, 0.0], [0.9, 0.1]], ["quantum qubit gate context", "grover shor algorithm context"], "quantum")
    _write_store(dbs["bio_general"], [[0.0, 1.0], [0.1, 0.9]], ["dna rna sequence context", "blast alignment protein context"], "bioinformatics")

    orch = MultiAgentRAGOrchestrator(db_dirs=dbs, n_clusters=2)
    stats = orch.build_index()
    assert stats["global_pool_size"] == 4

    out = orch.query("quantum qubit algorithm")
    assert "domain_prediction" in out
    assert len(out["top_chunks"]) <= 10
