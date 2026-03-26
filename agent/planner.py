"""Simple planner for selecting retrieval strategy."""


def route_query(query: str) -> str:
    """Route query to best-fit specialized node."""
    q = query.lower()
    if "equation" in q or "latex" in q:
        return "equation_node"
    if "hardware" in q or "qubit technology" in q:
        return "hardware_node"
    if "algorithm" in q or "shor" in q or "grover" in q or "vqe" in q or "qaoa" in q:
        return "algorithm_node"
    if "relationship" in q or "graph" in q:
        return "graph_node"
    return "general_rag"
