"""Neo4j graph utilities for domain-specific GraphRAG."""

from __future__ import annotations

import re
from dataclasses import dataclass

CONCEPT_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9\-]{2,})\b")


@dataclass
class GraphConfig:
    uri: str
    user: str
    password: str


class GraphNode:
    """Graph node manager with domain separation labels."""

    def __init__(self, config: GraphConfig):
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            raise RuntimeError("neo4j package is required for graph operations") from exc
        self.driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))

    def close(self) -> None:
        self.driver.close()

    def ensure_schema(self) -> None:
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE (c.name, c.domain) IS UNIQUE")

    def create_relation(self, term1: str, term2: str, source: str, domain: str, rel_type: str = "RELATED", weight: int = 1) -> None:
        """Create weighted relation in domain-specific concept space."""
        rel_type = rel_type if rel_type in {"RELATED", "TRANSCRIBED_TO"} else "RELATED"
        query = (
            "MERGE (a:Concept {name: $t1, domain: $domain}) "
            "MERGE (b:Concept {name: $t2, domain: $domain}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "ON CREATE SET r.weight = $weight, r.sources = [$source] "
            "ON MATCH SET r.weight = r.weight + $weight, r.sources = CASE WHEN $source IN r.sources THEN r.sources ELSE r.sources + $source END"
        )
        with self.driver.session() as session:
            session.run(query, t1=term1, t2=term2, source=source, weight=weight, domain=domain)

    def bulk_insert_relations(self, relations: list[dict]) -> None:
        for rel in relations:
            self.create_relation(
                rel["from"],
                rel["to"],
                rel.get("source", "unknown"),
                rel.get("domain", "quantum"),
                rel.get("rel_type", "RELATED"),
                rel.get("weight", 1),
            )

    def related(self, term: str, domain: str, limit: int = 5) -> list[str]:
        query = (
            "MATCH (:Concept {name:$term, domain:$domain})-[r]->(b:Concept {domain:$domain}) "
            "RETURN b.name AS name ORDER BY r.weight DESC LIMIT $limit"
        )
        with self.driver.session() as session:
            rows = session.run(query, term=term, domain=domain, limit=limit)
            return [r["name"] for r in rows]


def extract_concepts(text: str) -> list[str]:
    matches = CONCEPT_PATTERN.findall(text)
    block = {"The", "This", "That", "With", "From", "When", "Then"}
    filtered = [m for m in matches if m not in block]
    return list(dict.fromkeys(filtered))


def build_relations_from_record(record: dict) -> list[dict]:
    """Build concept pair edges from one chunk record based on domain."""
    domain = record.get("domain", "quantum")
    concepts = extract_concepts(record["chunk"])[:6]
    rels = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            rel_type = "RELATED"
            if domain == "bioinformatics" and {concepts[i].upper(), concepts[j].upper()} == {"DNA", "RNA"}:
                rel_type = "TRANSCRIBED_TO"
            rels.append(
                {
                    "from": concepts[i],
                    "to": concepts[j],
                    "source": record["source"],
                    "domain": domain,
                    "rel_type": rel_type,
                    "weight": 1,
                }
            )
    return rels
