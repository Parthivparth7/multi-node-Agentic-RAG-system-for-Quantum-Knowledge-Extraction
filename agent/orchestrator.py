"""Agent orchestrator for multi-domain vector + GraphRAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from agent.domain_classifier import DomainClassifier
from agent.planner import route_query
from config.settings import settings
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
    domain: str

    def render(self) -> str:
        related = ", ".join(self.related_concepts) if self.related_concepts else "None"
        return (
            f"1. Final Answer\n{self.answer}\n\n"
            f"2. Retrieved Context\n{self.retrieved_context}\n\n"
            f"3. Node Used\n{self.node_used}\n\n"
            f"4. Related Concepts (Graph)\n{related}\n\n"
            f"5. Source PDF\n{self.source_pdf}\n\n"
            f"6. Domain\n{self.domain}"
        )


class AgentOrchestrator:
    """Main orchestrator for domain detection, routing, and retrieval fusion."""

    def __init__(self):
        self.embedder = Embedder()
        self.domain_classifier = DomainClassifier(self.embedder)
        self._stores: dict[str, FaissStore] = {}

    def _get_store(self, node_name: str) -> FaissStore:
        if node_name not in self._stores:
            from config.settings import NODE_TO_DB_DIR

            self._stores[node_name] = FaissStore(NODE_TO_DB_DIR[node_name])
        return self._stores[node_name]

    def _fetch_vector_context(self, node_name: str, query: str, k: int = 4) -> list[dict]:
        store = self._get_store(node_name)
        qv = self.embedder.encode([query])[0]
        return store.search(qv, k=k)

    def _fetch_graph_concepts(self, seed_text: str, domain: str, limit: int = 5) -> list[str]:
        seeds = [w.strip(".,?!") for w in seed_text.split() if len(w) > 4][:3]
        if not seeds:
            return []
        cfg = GraphConfig(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            node = GraphNode(cfg)
            concepts = []
            for seed in seeds:
                concepts.extend(node.related(seed, domain=domain, limit=limit))
            node.close()
            return list(dict.fromkeys(concepts))[:limit]
        except Exception:
            return []

    @staticmethod
    def _history_hint(history: list[dict] | None, limit: int = 3) -> str:
        if not history:
            return ""
        turns = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-limit:]]
        return " | ".join(turns)

    def _run_single_domain(self, domain: str, query: str, history: list[dict] | None) -> OrchestratorResponse:
        node = route_query(query, domain=domain)
        top = self._fetch_vector_context(node, query, k=4)
        if not top:
            return OrchestratorResponse(
                answer="No indexed context found. Run ingestion and index build first.",
                retrieved_context="",
                node_used=node,
                related_concepts=[],
                source_pdf="N/A",
                domain=domain,
            )
        best = top[0]
        context = "\n---\n".join(item.get("chunk", "")[:400] for item in top)
        related = self._fetch_graph_concepts(best.get("chunk", ""), domain=domain)
        history_hint = self._history_hint(history)
        hint = f" Conversation hint: {history_hint}." if history_hint else ""
        answer = (
            f"Retrieved {domain} knowledge using node {node}."
            f"{hint} The answer is grounded in retrieved context and graph relations."
        )
        return OrchestratorResponse(
            answer=answer,
            retrieved_context=context,
            node_used=node,
            related_concepts=related,
            source_pdf=best.get("source", "unknown"),
            domain=domain,
        )

    def run(
        self,
        query: str,
        node_override: str | None = None,
        vector_override: str | None = None,
        history: list[dict] | None = None,
    ) -> OrchestratorResponse:
        detected = self.domain_classifier.classify(query)
        if detected == "cross":
            q = self._run_single_domain("quantum", query, history)
            b = self._run_single_domain("bioinformatics", query, history)
            return OrchestratorResponse(
                answer=f"Cross-domain fusion completed. Quantum + Bio insights merged.\n\nQ: {q.answer}\nB: {b.answer}",
                retrieved_context=f"[Quantum]\n{q.retrieved_context}\n\n[Bioinformatics]\n{b.retrieved_context}",
                node_used=f"{q.node_used} + {b.node_used}",
                related_concepts=list(dict.fromkeys(q.related_concepts + b.related_concepts)),
                source_pdf=f"{q.source_pdf}; {b.source_pdf}",
                domain="cross",
            )

        domain = detected
        if node_override and node_override != "auto":
            node = node_override
            top = self._fetch_vector_context(node, query, k=4)
            if not top:
                return OrchestratorResponse("No indexed context found.", "", node, [], "N/A", domain)
            best = top[0]
            return OrchestratorResponse(
                answer=f"Retrieved {domain} knowledge via override node {node}.",
                retrieved_context="\n---\n".join(item.get("chunk", "")[:400] for item in top),
                node_used=node,
                related_concepts=self._fetch_graph_concepts(best.get("chunk", ""), domain=domain),
                source_pdf=best.get("source", "unknown"),
                domain=domain,
            )

        return self._run_single_domain(domain, query, history)

    def stream(
        self,
        query: str,
        node_override: str | None = None,
        vector_override: str | None = None,
        history: list[dict] | None = None,
        chunk_size: int = 18,
    ) -> tuple[Iterator[str], OrchestratorResponse]:
        result = self.run(query, node_override=node_override, vector_override=vector_override, history=history)
        return self.stream_text(result.answer, chunk_size=chunk_size), result

    @staticmethod
    def stream_text(text: str, chunk_size: int = 18) -> Iterator[str]:
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
