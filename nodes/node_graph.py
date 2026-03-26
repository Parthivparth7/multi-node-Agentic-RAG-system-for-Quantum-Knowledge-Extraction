"""Neo4j graph utilities for concept relationships and GraphRAG."""

from __future__ import annotations

import re
from dataclasses import dataclass

CONCEPT_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9\-]{2,})\b")


@dataclass
class GraphConfig:
    """Connection settings for Neo4j."""

    uri: str
    user: str
    password: str


class GraphNode:
    """Graph node manager for concept relationship creation/query."""

    def __init__(self, config: GraphConfig):
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            raise RuntimeError("neo4j package is required for graph operations") from exc
        self.driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))

    def close(self) -> None:
        """Close driver session."""
        self.driver.close()

    def ensure_schema(self) -> None:
        """Create indexes/constraints for graph operations."""
        query = "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE"
        with self.driver.session() as session:
            session.run(query)

    def create_relation(self, term1: str, term2: str, source: str, weight: int = 1) -> None:
        """Create or update weighted Concept->RELATED->Concept relation."""
        query = (
            "MERGE (a:Concept {name: $t1}) "
            "MERGE (b:Concept {name: $t2}) "
            "MERGE (a)-[r:RELATED]->(b) "
            "ON CREATE SET r.weight = $weight, r.sources = [$source] "
            "ON MATCH SET r.weight = r.weight + $weight, r.sources = CASE WHEN $source IN r.sources THEN r.sources ELSE r.sources + $source END"
        )
        with self.driver.session() as session:
            session.run(query, t1=term1, t2=term2, source=source, weight=weight)

    def bulk_insert_relations(self, relations: list[dict]) -> None:
        """Batch insert weighted relations."""
        for rel in relations:
            self.create_relation(rel["from"], rel["to"], rel.get("source", "unknown"), rel.get("weight", 1))

    def related(self, term: str, limit: int = 5) -> list[str]:
        """Fetch related concepts for a term."""
        query = (
            "MATCH (:Concept {name:$term})-[r:RELATED]->(b:Concept) "
            "RETURN b.name AS name ORDER BY r.weight DESC LIMIT $limit"
        )
        with self.driver.session() as session:
            rows = session.run(query, term=term, limit=limit)
            return [r["name"] for r in rows]


def extract_concepts(text: str) -> list[str]:
    """Extract coarse concept candidates from text."""
    matches = CONCEPT_PATTERN.findall(text)
    block = {"The", "This", "That", "With", "From", "When", "Then"}
    filtered = [m for m in matches if m not in block]
    return list(dict.fromkeys(filtered))


def build_relations_from_record(record: dict) -> list[dict]:
    """Build concept pair edges from one chunk record."""
    concepts = extract_concepts(record["chunk"])[:6]
    rels: list[dict] = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            rels.append({"from": concepts[i], "to": concepts[j], "source": record["source"], "weight": 1})
    return rels
