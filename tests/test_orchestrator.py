from agent.orchestrator import AgentOrchestrator


class DummyOrchestrator(AgentOrchestrator):
    def _fetch_vector_context(self, node: str, query: str, k: int = 4):
        return [{"chunk": "Grover gives quadratic speedup.", "source": "qc.pdf"}]

    def _fetch_graph_concepts(self, seed_text: str, limit: int = 5):
        return ["Amplitude Amplification", "Oracle"]


def test_orchestrator_response_shape():
    orch = DummyOrchestrator()
    history = [{"role": "user", "content": "previous question about qubits"}]
    result = orch.run("Explain Grover algorithm", history=history)
    text = result.render()
    assert "1. Final Answer" in text
    assert result.node_used == "algorithm"
    assert result.source_pdf == "qc.pdf"
    assert "Conversation hint" in result.answer


def test_stream_text():
    orch = DummyOrchestrator()
    chunks = list(orch.stream_text("abcdef", chunk_size=2))
    assert chunks == ["ab", "cd", "ef"]


def test_stream_interface():
    orch = DummyOrchestrator()
    stream_iter, result = orch.stream("Explain Grover", history=[{"role": "user", "content": "hi"}])
    assert result.source_pdf == "qc.pdf"
    assert "".join(list(stream_iter))
