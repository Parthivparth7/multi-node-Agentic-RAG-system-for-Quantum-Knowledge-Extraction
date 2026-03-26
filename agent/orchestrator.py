"""Agent orchestrator: combines router + retrieval output format."""

from __future__ import annotations

from agent.planner import route_query


def format_response(answer: str, context: str, related: list[str], source: str) -> str:
    """Render response in mandatory 4-section format."""
    related_text = ", ".join(related) if related else "None"
    return (
        f"1. Answer\n{answer}\n\n"
        f"2. Supporting Context\n{context}\n\n"
        f"3. Related Concepts\n{related_text}\n\n"
        f"4. Source (PDF name)\n{source}"
    )


def orchestrate(query: str) -> str:
    """Placeholder orchestration demonstrating route behavior."""
    node = route_query(query)
    return format_response(
        answer=f"Query routed to {node}. Integrate node-specific retrieval and generation here.",
        context="No retrieval executed in scaffold mode.",
        related=[],
        source="N/A",
    )
