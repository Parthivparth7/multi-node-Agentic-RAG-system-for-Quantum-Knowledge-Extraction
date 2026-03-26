from pathlib import Path

from rag.embedder import Embedder
from rag.retriever import FaissStore


def test_embedder_outputs_vectors():
    emb = Embedder()
    vectors = emb.encode(["qubit", "entanglement"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0


def test_faiss_store_roundtrip_with_rerank_and_threshold(tmp_path: Path):
    store = FaissStore(tmp_path / "faiss")
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    metadata = [{"chunk": "a"}, {"chunk": "b"}, {"chunk": "c"}, {"chunk": "d"}]
    store.build(vectors, metadata)
    store.save()

    loaded = FaissStore(tmp_path / "faiss")
    loaded.load()
    results = loaded.search([1.0, 0.0], k=3, candidate_k=10, score_threshold=0.3)
    assert results
    assert len(results) <= 3
    assert results[0]["chunk"] in {"a", "b"}
