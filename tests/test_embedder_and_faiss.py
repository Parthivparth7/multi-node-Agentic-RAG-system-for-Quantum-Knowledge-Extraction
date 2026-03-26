from pathlib import Path

from rag.embedder import Embedder
from rag.retriever import FaissStore


def test_embedder_outputs_vectors():
    emb = Embedder()
    vectors = emb.encode(["qubit", "entanglement"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0


def test_faiss_store_roundtrip(tmp_path: Path):
    store = FaissStore(tmp_path / "faiss")
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    metadata = [{"chunk": "a"}, {"chunk": "b"}, {"chunk": "c"}]
    store.build(vectors, metadata)
    store.save()

    loaded = FaissStore(tmp_path / "faiss")
    loaded.load()
    results = loaded.search([1.0, 0.0], k=2)
    assert results
    assert results[0]["chunk"] in {"a", "b"}
