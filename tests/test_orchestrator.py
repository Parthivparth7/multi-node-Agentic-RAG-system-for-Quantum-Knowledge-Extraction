from agent.orchestrator import AgentOrchestrator


class DummyClassifier:
    def __init__(self, mode):
        self.mode = mode

    def classify(self, query: str):
        return self.mode

    def classify_with_confidence(self, query: str):
        return {"domain": self.mode, "confidence": 0.9, "scores": {"quantum": 0.9, "bioinformatics": 0.1}}


class DummyOrchestrator(AgentOrchestrator):
    def __init__(self, mode="quantum", score=0.8):
        super().__init__()
        self.domain_classifier = DummyClassifier(mode)
        self.score = score

    def _fetch_vector_context(self, node_name: str, query: str, k: int = 3):
        domain = "quantum" if node_name.startswith("quantum") else "bioinformatics"
        return [{"chunk": f"{domain} chunk", "source": f"{domain}.pdf", "domain": domain, "score": self.score}]

    def _fetch_graph_concepts(self, seed_text: str, domain: str, limit: int = 5):
        return [f"{domain}_concept"]


def test_orchestrator_grounded_answer():
    orch = DummyOrchestrator(mode="quantum", score=0.8)
    result = orch.run("Explain Grover algorithm", history=[{"role": "user", "content": "prev"}])
    assert result.domain.startswith("quantum")
    assert result.answer.startswith("Based on retrieved context:")


def test_orchestrator_low_confidence_reject():
    orch = DummyOrchestrator(mode="quantum", score=0.1)
    result = orch.run("Explain Grover algorithm")
    assert result.answer == "No reliable answer found in indexed data"


def test_orchestrator_cross_domain():
    orch = DummyOrchestrator(mode="cross", score=0.8)
    result = orch.run("quantum genomics")
    assert result.domain.startswith("cross")
    assert "+" in result.node_used


def test_stream_interface():
    orch = DummyOrchestrator(mode="quantum", score=0.8)
    stream_iter, result = orch.stream("Explain Grover")
    assert result.source_pdf.endswith(".pdf")
    assert "".join(list(stream_iter))
