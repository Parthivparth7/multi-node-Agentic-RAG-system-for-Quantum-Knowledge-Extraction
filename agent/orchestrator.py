"""Agent orchestrator for multi-node vector + GraphRAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from agent.planner import route_query
from config.settings import NODE_TO_DB_DIR, settings
from nodes.node_graph import GraphConfig, GraphNode
from rag.embedder import Embedder
from rag.retriever import FaissStore


@dataclass
class OrchestratorResponse:
    answer: str
    retrieved_context: str
    node_used: str
    related_concepts: list[str]
    source_pdf: str

    def render(self) -> str:
        """Render strict response format."""
        related = ", ".join(self.related_concepts) if self.related_concepts else "None"
        return (
            f"1. Final Answer\n{self.answer}\n\n"
            f"2. Retrieved Context\n{self.retrieved_context}\n\n"
            f"3. Node Used\n{self.node_used}\n\n"
            f"4. Related Concepts (Graph)\n{related}\n\n"
            f"5. Source PDF\n{self.source_pdf}"
        )


class AgentOrchestrator:
    """Main orchestrator for routing and retrieval."""

    def __init__(self):
        self.embedder = Embedder()

    def _fetch_vector_context(self, node: str, query: str, k: int = 4) -> list[dict]:
        store = FaissStore(NODE_TO_DB_DIR[node])
        qv = self.embedder.encode([query])[0]
        return store.search(qv, k=k)

    def _fetch_graph_concepts(self, seed_text: str, limit: int = 5) -> list[str]:
        seeds = [w.strip(".,?!") for w in seed_text.split() if len(w) > 4][:3]
        if not seeds:
            return []
        cfg = GraphConfig(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            node = GraphNode(cfg)
            concepts = []
            for seed in seeds:
                concepts.extend(node.related(seed, limit=limit))
            node.close()
            return list(dict.fromkeys(concepts))[:limit]
        except Exception:
            return []

    def run(self, query: str, node_override: str | None = None, vector_override: str | None = None) -> OrchestratorResponse:
        """Run end-to-end retrieval and return structured response."""
        routed = node_override if node_override and node_override != "auto" else route_query(query)
        db_node = vector_override if vector_override and vector_override != "auto" else routed
        top = self._fetch_vector_context(db_node, query, k=4)

        if not top:
            return OrchestratorResponse(
                answer="No indexed context found. Run ingestion and index build first.",
                retrieved_context="",
                node_used=routed,
                related_concepts=[],
                source_pdf="N/A",
            )

        best = top[0]
        context = "\n---\n".join(item.get("chunk", "")[:400] for item in top)
        related = self._fetch_graph_concepts(best.get("chunk", ""))

        answer = (
            "Based on retrieved quantum corpus context, this response is grounded in the top matching chunks. "
            "For deeper validation, inspect the cited source chunk and related concepts."
        )
        return OrchestratorResponse(
            answer=answer,
            retrieved_context=context,
            node_used=routed,
            related_concepts=related,
            source_pdf=best.get("source", "unknown"),
        )
