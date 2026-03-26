from agent.domain_classifier import DomainClassifier


class DummyEmbedder:
    def encode(self, texts):
        out = []
        for t in texts:
            tl = t.lower()
            out.append([
                1.0 if "qubit" in tl or "quantum" in tl else 0.0,
                1.0 if "gene" in tl or "dna" in tl or "bio" in tl else 0.0,
            ])
        return out


def test_domain_classifier_quantum_and_bio():
    clf = DomainClassifier(embedder=DummyEmbedder())
    assert clf.classify("quantum qubit gate") == "quantum"
    assert clf.classify("dna gene sequence") == "bioinformatics"
    assert clf.classify("quantum and gene") == "cross"


def test_classifier_confidence_schema():
    clf = DomainClassifier(embedder=DummyEmbedder())
    out = clf.classify_with_confidence("quantum qubit gate")
    assert set(out.keys()) == {"domain", "confidence", "scores"}
    assert isinstance(out["confidence"], float)
