from nodes.node_algorithms import annotate_algorithms
from nodes.node_common_terms import annotate_chunks_with_terms
from nodes.node_equations import annotate_equations
from nodes.node_hardware import annotate_hardware


def _records():
    return [
        {
            "id": "1",
            "source": "book.pdf",
            "page": 1,
            "chunk": "Grover algorithm and Shor algorithm. Hamiltonian $H=Z$ for superconducting qubits.",
        }
    ]


def test_node_annotations():
    records = _records()
    assert annotate_chunks_with_terms(records)[0]["terms"]
    assert annotate_equations(records)[0]["equations"]
    assert annotate_hardware(records)[0]["hardware_tags"]
    assert annotate_algorithms(records)[0]["algorithms"]
