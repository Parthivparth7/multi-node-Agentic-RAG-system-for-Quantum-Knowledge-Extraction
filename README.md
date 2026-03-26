# Black Kite — Multi-Domain Knowledge Retrieval Agent

Offline-first Agentic RAG for **Quantum Computing** and **Bioinformatics** books.

## Data Layout

- Quantum PDFs: `data/raw_pdfs/Quantum-Computing-Books/`
- Bioinformatics PDFs: `data/raw_pdfs/Bioinformatics-Books/`

## Core Flow

1. Ingest + chunk with domain tagging (`scripts/ingest.py`)
2. Build separate FAISS indexes per domain/node (`scripts/build_vector_db.py`)
3. Build domain-aware Neo4j graph (`scripts/build_graph.py`)
4. Query through multi-domain orchestrator (`agent/orchestrator.py`)

## Domain-separated FAISS stores

### Quantum
- `faiss_quantum_general`
- `faiss_quantum_equations`
- `faiss_quantum_hardware`
- `faiss_quantum_algorithms`

### Bioinformatics
- `faiss_bio_general`
- `faiss_bio_terms`
- `faiss_bio_algorithms`

## Domain classifier

`agent/domain_classifier.py` uses embedding similarity against domain prototypes:
- `quantum`
- `bioinformatics`
- `cross` (for mixed-domain queries)

## Run

```bash
python -m scripts.ingest
python -m scripts.build_vector_db
python -m scripts.build_graph
pytest -q
streamlit run ui/streamlit_app.py
```


Retrieval behavior:
- top-10 candidate retrieval with cosine reranking
- top-3 returned after score threshold filtering (default 0.3)
- if top score below threshold, system returns: "No reliable answer found in indexed data"
