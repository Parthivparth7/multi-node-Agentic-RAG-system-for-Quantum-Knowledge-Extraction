from agent.orchestrator import AgentOrchestrator


class DummyClassifier:
    def __init__(self, mode):
        self.mode = mode

    def classify(self, query: str):
        return self.mode


class DummyOrchestrator(AgentOrchestrator):
    def __init__(self, mode="quantum"):
        super().__init__()
        self.domain_classifier = DummyClassifier(mode)

    def _fetch_vector_context(self, node_name: str, query: str, k: int = 4):
        domain = "quantum" if node_name.startswith("quantum") else "bioinformatics"
        return [{"chunk": f"{domain} chunk", "source": f"{domain}.pdf", "domain": domain}]

    def _fetch_graph_concepts(self, seed_text: str, domain: str, limit: int = 5):
        return [f"{domain}_concept"]


def test_orchestrator_quantum_route():
    orch = DummyOrchestrator(mode="quantum")
    result = orch.run("Explain Grover algorithm", history=[{"role": "user", "content": "prev"}])
    assert result.domain == "quantum"
    assert result.node_used.startswith("quantum")


def test_orchestrator_cross_domain():
    orch = DummyOrchestrator(mode="cross")
    result = orch.run("quantum genomics")
    assert result.domain == "cross"
    assert "+" in result.node_used


def test_stream_interface():
    orch = DummyOrchestrator(mode="quantum")
    stream_iter, result = orch.stream("Explain Grover")
    assert result.source_pdf.endswith(".pdf")
    assert "".join(list(stream_iter))
