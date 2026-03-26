"""Neo4j graph utilities for concept relationships."""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import GraphDatabase


@dataclass
class GraphConfig:
    """Connection settings for Neo4j."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"


class GraphNode:
    """Graph node manager for concept relationship creation/query."""

    def __init__(self, config: GraphConfig):
        self.driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))

    def close(self) -> None:
        """Close driver session."""
        self.driver.close()

    def create_relation(self, term1: str, term2: str) -> None:
        """Create a Concept->RELATED->Concept relation."""
        query = (
            "MERGE (a:Concept {name: $t1}) "
            "MERGE (b:Concept {name: $t2}) "
            "MERGE (a)-[:RELATED]->(b)"
        )
        with self.driver.session() as session:
            session.run(query, t1=term1, t2=term2)

    def related(self, term: str, limit: int = 5) -> list[str]:
        """Fetch related concepts for a term."""
        query = (
            "MATCH (:Concept {name:$term})-[:RELATED]->(b:Concept) "
            "RETURN b.name AS name LIMIT $limit"
        )
        with self.driver.session() as session:
            rows = session.run(query, term=term, limit=limit)
            return [r["name"] for r in rows]
