from agent.orchestrator import AgentOrchestrator


class DummyOrchestrator(AgentOrchestrator):
    def _fetch_vector_context(self, node: str, query: str, k: int = 4):
        return [{"chunk": "Grover gives quadratic speedup.", "source": "qc.pdf"}]

    def _fetch_graph_concepts(self, seed_text: str, limit: int = 5):
        return ["Amplitude Amplification", "Oracle"]


def test_orchestrator_response_shape():
    orch = DummyOrchestrator()
    result = orch.run("Explain Grover algorithm")
    text = result.render()
    assert "1. Final Answer" in text
    assert result.node_used == "algorithm"
    assert result.source_pdf == "qc.pdf"
