"""Build Neo4j concept graph from chunk records."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import settings
from nodes.node_graph import GraphConfig, GraphNode, build_relations_from_record

logger = logging.getLogger(__name__)


def load_chunks(path: Path = settings.chunks_file) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(chunks_file: Path = settings.chunks_file) -> int:
    """Build graph edges from chunk corpus."""
    records = load_chunks(chunks_file)
    cfg = GraphConfig(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    graph = GraphNode(cfg)
    graph.ensure_schema()

    inserted = 0
    for rec in records:
        rels = build_relations_from_record(rec)
        if rels:
            graph.bulk_insert_relations(rels)
            inserted += len(rels)

    graph.close()
    logger.info("Inserted %s relations", inserted)
    return inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = run()
    print(f"Inserted relations: {count}")
