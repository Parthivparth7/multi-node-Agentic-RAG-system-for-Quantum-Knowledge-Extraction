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

    def _fetch_vector_context(self, node_name: str, query: str, k: int = 3) -> list[dict]:
        store = self._get_store(node_name)
        qv = self.embedder.encode([query])[0]
        return store.search(qv, k=k, candidate_k=10, score_threshold=0.3)

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
    def _history_hint(history: list[dict] | None, limit: int = 2) -> str:
        if not history:
            return ""
        turns = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-limit:]]
        return " | ".join(turns)

    @staticmethod
    def _grounded_answer(top_chunks: list[dict], graph_concepts: list[str], history_hint: str) -> str:
        """Build answer strictly from retrieved context and graph concepts."""
        context_bits = [c.get("chunk", "")[:220] for c in top_chunks[:3] if c.get("chunk")]
        if not context_bits:
            return "No reliable answer found in indexed data"
        merged_context = " ".join(context_bits)
        graph_text = f" Related graph concepts: {', '.join(graph_concepts)}." if graph_concepts else ""
        history_text = f" Prior context: {history_hint}." if history_hint else ""
        return f"Based on retrieved context: {merged_context}{graph_text}{history_text}"

    def _run_single_domain(self, domain: str, query: str, history: list[dict] | None) -> OrchestratorResponse:
        node = route_query(query, domain=domain)
        top = self._fetch_vector_context(node, query, k=3)

        if not top:
            return OrchestratorResponse("No reliable answer found in indexed data", "", node, [], "N/A", domain)

        best = top[0]
        if float(best.get("score", 0.0)) < 0.3:
            return OrchestratorResponse("No reliable answer found in indexed data", "", node, [], best.get("source", "N/A"), domain)

        context = "\n---\n".join(item.get("chunk", "")[:400] for item in top)
        related = self._fetch_graph_concepts(best.get("chunk", ""), domain=domain)
        answer = self._grounded_answer(top, related, self._history_hint(history))

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
        cls = self.domain_classifier.classify_with_confidence(query)
        detected = cls["domain"]

        if detected == "cross":
            q = self._run_single_domain("quantum", query, history)
            b = self._run_single_domain("bioinformatics", query, history)
            answer = self._grounded_answer(
                [{"chunk": q.retrieved_context}, {"chunk": b.retrieved_context}],
                list(dict.fromkeys(q.related_concepts + b.related_concepts)),
                self._history_hint(history),
            )
            return OrchestratorResponse(
                answer=answer,
                retrieved_context=f"[Quantum]\n{q.retrieved_context}\n\n[Bioinformatics]\n{b.retrieved_context}",
                node_used=f"{q.node_used} + {b.node_used}",
                related_concepts=list(dict.fromkeys(q.related_concepts + b.related_concepts)),
                source_pdf=f"{q.source_pdf}; {b.source_pdf}",
                domain=f"cross (confidence={cls['confidence']})",
            )

        domain = detected
        if node_override and node_override != "auto":
            node = node_override
            top = self._fetch_vector_context(node, query, k=3)
            if not top or float(top[0].get("score", 0.0)) < 0.3:
                return OrchestratorResponse("No reliable answer found in indexed data", "", node, [], "N/A", domain)
            best = top[0]
            related = self._fetch_graph_concepts(best.get("chunk", ""), domain=domain)
            return OrchestratorResponse(
                answer=self._grounded_answer(top, related, self._history_hint(history)),
                retrieved_context="\n---\n".join(item.get("chunk", "")[:400] for item in top),
                node_used=node,
                related_concepts=related,
                source_pdf=best.get("source", "unknown"),
                domain=f"{domain} (confidence={cls['confidence']})",
            )

        out = self._run_single_domain(domain, query, history)
        out.domain = f"{domain} (confidence={cls['confidence']})"
        return out

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
