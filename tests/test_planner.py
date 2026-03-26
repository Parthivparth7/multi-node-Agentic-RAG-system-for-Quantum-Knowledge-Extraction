from agent.planner import route_query


def test_route_query():
    assert route_query("explain this equation") == "equation_node"
    assert route_query("latest hardware roadmap") == "hardware_node"
    assert route_query("how does grover algorithm work") == "algorithm_node"
    assert route_query("show graph relationship") == "graph_node"
    assert route_query("what is a qubit") == "general_rag"
