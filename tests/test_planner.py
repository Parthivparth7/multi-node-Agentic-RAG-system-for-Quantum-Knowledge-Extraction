from agent.planner import route_query


def test_route_query_quantum():
    assert route_query("explain this equation", domain="quantum") == "quantum_equation"
    assert route_query("superconducting hardware roadmap", domain="quantum") == "quantum_hardware"
    assert route_query("how does grover algorithm work", domain="quantum") == "quantum_algorithm"
    assert route_query("what is a qubit", domain="quantum") == "quantum_general"


def test_route_query_bio():
    assert route_query("run BLAST alignment", domain="bioinformatics") == "bio_algorithm"
    assert route_query("gene and protein function", domain="bioinformatics") == "bio_terms"
    assert route_query("omics overview", domain="bioinformatics") == "bio_general"
