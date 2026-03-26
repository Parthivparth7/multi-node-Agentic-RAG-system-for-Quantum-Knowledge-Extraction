from nodes.node_graph import build_relations_from_record, extract_concepts


def test_extract_concepts_and_relations():
    rec = {
        "source": "x.pdf",
        "chunk": "Qubit Entanglement BellState Grover Quantum Fourier Transform are linked.",
    }
    concepts = extract_concepts(rec["chunk"])
    assert "Qubit" in concepts
    rels = build_relations_from_record(rec)
    assert rels
    assert {"from", "to", "source", "weight"}.issubset(rels[0].keys())
