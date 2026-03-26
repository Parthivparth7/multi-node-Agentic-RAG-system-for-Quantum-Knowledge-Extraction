from agent.planner import route_query


def test_route_query():
    assert route_query("explain this equation") == "equation"
    assert route_query("superconducting hardware roadmap") == "hardware"
    assert route_query("how does grover algorithm work") == "algorithm"
    assert route_query("important quantum terms") == "common_terms"
    assert route_query("what is a qubit") == "general"
