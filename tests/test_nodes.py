from nodes.node_algorithms import annotate_algorithms
from nodes.node_bio import annotate_bio_algorithms, annotate_bio_terms
from nodes.node_equations import annotate_equations
from nodes.node_hardware import annotate_hardware


def test_quantum_node_annotations():
    records = [
        {
            "id": "1",
            "domain": "quantum",
            "source": "book.pdf",
            "page": 1,
            "chunk": "Grover algorithm and Shor algorithm. Hamiltonian $H=Z$ for superconducting qubits.",
        }
    ]
    assert annotate_equations(records)[0]["equations"]
    assert annotate_hardware(records)[0]["hardware_tags"]
    assert annotate_algorithms(records)[0]["algorithms"]


def test_bio_node_annotations():
    records = [
        {
            "id": "2",
            "domain": "bioinformatics",
            "source": "bio.pdf",
            "page": 2,
            "chunk": "DNA and RNA alignment using BLAST helps gene analysis.",
        }
    ]
    assert annotate_bio_terms(records)[0]["bio_entities"]
    assert annotate_bio_algorithms(records)[0]["bio_algorithms"]
