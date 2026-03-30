"""Multi-agent clustered RAG over multiple FAISS databases.

This script adds a scalable retrieval layer:
1) loads multiple vector stores,
2) creates a global vector pool,
3) clusters vectors via MiniBatchKMeans,
4) routes/classifies query by cluster distribution,
5) performs cluster-first semantic search,
6) expands context windows,
7) reranks and filters noisy chunks,
8) returns top high-quality chunks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:  # pragma: no cover
    MiniBatchKMeans = None

from config.settings import NODE_TO_DB_DIR
from rag.embedder import Embedder


@dataclass
class ChunkRecord:
    vector: list[float]
    meta: dict


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _is_noisy_chunk(text: str, min_len: int = 40, numeric_ratio_threshold: float = 0.35) -> bool:
    if len(text.strip()) < min_len:
        return True
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    numeric = sum(c.isdigit() for c in chars)
    return (numeric / max(len(chars), 1)) > numeric_ratio_threshold


class VectorDBLoaderAgent:
    """Loads vectors and metadata from multiple FAISS directories."""

    def __init__(self, db_dirs: dict[str, Path]):
        self.db_dirs = db_dirs

    def load(self) -> dict[str, list[ChunkRecord]]:
        stores: dict[str, list[ChunkRecord]] = {}
        for name, db_dir in self.db_dirs.items():
            vec_path = db_dir / "vectors.json"
            meta_path = db_dir / "metadata.jsonl"
            if not vec_path.exists() or not meta_path.exists():
                stores[name] = []
                continue

            with vec_path.open("r", encoding="utf-8") as f:
                vectors = json.load(f)
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = [json.loads(line) for line in f]

            records = []
            for v, m in zip(vectors, metadata):
                m = dict(m)
                m["db_name"] = name
                records.append(ChunkRecord(vector=list(map(float, v)), meta=m))
            stores[name] = records
        return stores


class GlobalPoolAgent:
    """Combines all db vectors into one global pool."""

    def combine(self, stores: dict[str, list[ChunkRecord]]) -> list[ChunkRecord]:
        pool: list[ChunkRecord] = []
        for records in stores.values():
            pool.extend(records)
        return pool


class ClusteringAgent:
    """Creates MiniBatchKMeans clusters and cluster->vector mapping."""

    def __init__(self, n_clusters: int = 100, batch_size: int = 2048, random_state: int = 42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        self.kmeans = None
        self.cluster_to_indices: dict[int, list[int]] = {}
        self.domain_cluster_hist: dict[str, dict[int, int]] = {}

    def fit(self, pool: list[ChunkRecord]) -> None:
        if not pool:
            self.cluster_to_indices = {}
            self.domain_cluster_hist = {}
            return

        effective_clusters = min(self.n_clusters, max(2, len(pool) // 5))
        vectors = [r.vector for r in pool]

        if MiniBatchKMeans is None or np is None:
            # fallback: single pseudo cluster
            self.cluster_to_indices = {0: list(range(len(pool)))}
            self.domain_cluster_hist = self._domain_hist(pool, [0] * len(pool))
            self.kmeans = None
            return

        arr = np.array(vectors, dtype=np.float32)
        self.kmeans = MiniBatchKMeans(
            n_clusters=effective_clusters,
            batch_size=min(self.batch_size, len(pool)),
            random_state=self.random_state,
            n_init="auto",
        )
        labels = self.kmeans.fit_predict(arr)

        mapping: dict[int, list[int]] = {}
        for idx, label in enumerate(labels.tolist()):
            mapping.setdefault(int(label), []).append(idx)
        self.cluster_to_indices = mapping
        self.domain_cluster_hist = self._domain_hist(pool, labels.tolist())

    @staticmethod
    def _domain_hist(pool: list[ChunkRecord], labels: list[int]) -> dict[str, dict[int, int]]:
        hist: dict[str, dict[int, int]] = {}
        for rec, label in zip(pool, labels):
            domain = rec.meta.get("domain", "unknown")
            hist.setdefault(domain, {})
            hist[domain][label] = hist[domain].get(label, 0) + 1
        return hist

    def nearest_clusters(self, query_vec: list[float], top_c: int = 5) -> list[int]:
        if not self.cluster_to_indices:
            return []
        if self.kmeans is None or np is None:
            return list(self.cluster_to_indices.keys())[:top_c]

        centers = self.kmeans.cluster_centers_.tolist()
        scored = [(_cos(query_vec, c), idx) for idx, c in enumerate(centers)]
        scored.sort(reverse=True)
        return [idx for _, idx in scored[:top_c]]


class DomainClassifierByClusters:
    """Classifies query domain by nearest-cluster distribution."""

    def __init__(self, clustering: ClusteringAgent):
        self.clustering = clustering

    def classify(self, query_vec: list[float], top_c: int = 5) -> dict:
        clusters = self.clustering.nearest_clusters(query_vec, top_c=top_c)
        domain_scores: dict[str, int] = {}
        for domain, hist in self.clustering.domain_cluster_hist.items():
            domain_scores[domain] = sum(hist.get(c, 0) for c in clusters)

        if not domain_scores:
            return {"domain": "unknown", "confidence": 0.0, "clusters": clusters}

        ranked = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_count = ranked[0]
        total = sum(domain_scores.values()) or 1
        confidence = best_count / total

        # mixed if top-2 are close
        if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= max(1, int(0.1 * total)):
            return {"domain": "cross", "confidence": confidence, "clusters": clusters, "scores": domain_scores}

        return {"domain": best_domain, "confidence": confidence, "clusters": clusters, "scores": domain_scores}


class SemanticSearchAgent:
    """Cluster-first semantic search and reranking."""

    def __init__(self, pooling: list[ChunkRecord], clustering: ClusteringAgent):
        self.pool = pooling
        self.clustering = clustering

    def search(self, query_vec: list[float], top_k: int = 20, top_c: int = 5) -> list[int]:
        clusters = self.clustering.nearest_clusters(query_vec, top_c=top_c)
        candidates: list[int] = []
        for c in clusters:
            candidates.extend(self.clustering.cluster_to_indices.get(c, []))

        # dedupe
        candidates = list(dict.fromkeys(candidates))
        scored = [(_cos(query_vec, self.pool[i].vector), i) for i in candidates]
        scored.sort(reverse=True)
        return [i for _, i in scored[:top_k]]


class ContextExpansionAgent:
    """Expands each hit to neighboring indices (+/- window) and reranks."""

    def __init__(self, pool: list[ChunkRecord]):
        self.pool = pool

    def expand(self, hit_indices: Iterable[int], before: int = 50, after: int = 150) -> list[int]:
        expanded = set()
        n = len(self.pool)
        for idx in hit_indices:
            s = max(0, idx - before)
            e = min(n - 1, idx + after)
            expanded.update(range(s, e + 1))
        return sorted(expanded)

    def rerank_and_filter(self, query_vec: list[float], expanded_indices: list[int], top_n: int = 10) -> list[dict]:
        scored = []
        for i in expanded_indices:
            rec = self.pool[i]
            text = rec.meta.get("chunk", "")
            if _is_noisy_chunk(text):
                continue
            sim = _cos(query_vec, rec.vector)
            scored.append((sim, i))

        scored.sort(reverse=True)
        out = []
        for sim, idx in scored[:top_n]:
            row = dict(self.pool[idx].meta)
            row["score"] = float(sim)
            row["global_idx"] = idx
            out.append(row)
        return out


class MultiAgentRAGOrchestrator:
    """Connects all agents for large-scale clustered retrieval."""

    def __init__(self, db_dirs: dict[str, Path] | None = None, n_clusters: int = 100):
        self.db_dirs = db_dirs or NODE_TO_DB_DIR
        self.embedder = Embedder()

        self.loader = VectorDBLoaderAgent(self.db_dirs)
        self.pooler = GlobalPoolAgent()
        self.clusterer = ClusteringAgent(n_clusters=n_clusters)

        self.pool: list[ChunkRecord] = []
        self.domain_classifier = None
        self.search_agent = None
        self.expand_agent = None

    def build_index(self) -> dict:
        stores = self.loader.load()
        self.pool = self.pooler.combine(stores)
        self.clusterer.fit(self.pool)

        self.domain_classifier = DomainClassifierByClusters(self.clusterer)
        self.search_agent = SemanticSearchAgent(self.pool, self.clusterer)
        self.expand_agent = ContextExpansionAgent(self.pool)

        return {
            "stores_loaded": {k: len(v) for k, v in stores.items()},
            "global_pool_size": len(self.pool),
            "clusters": len(self.clusterer.cluster_to_indices),
        }

    def query(self, text: str) -> dict:
        if self.search_agent is None:
            self.build_index()

        qv = self.embedder.encode([text])[0]
        cls = self.domain_classifier.classify(qv, top_c=5)
        hits = self.search_agent.search(qv, top_k=20, top_c=5)
        expanded = self.expand_agent.expand(hits, before=50, after=150)
        top_chunks = self.expand_agent.rerank_and_filter(qv, expanded, top_n=10)

        return {
            "domain_prediction": cls,
            "initial_hits": len(hits),
            "expanded_candidates": len(expanded),
            "top_chunks": top_chunks,
        }


if __name__ == "__main__":
    orchestrator = MultiAgentRAGOrchestrator()
    stats = orchestrator.build_index()
    print("Index stats:", stats)
    sample = orchestrator.query("quantum algorithms for genomics and sequence optimization")
    print("Prediction:", sample["domain_prediction"])
    print("Top chunks:", len(sample["top_chunks"]))
