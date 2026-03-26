# Black Kite QKRA — Multi-node Agentic RAG for Quantum Knowledge

Offline-first production pipeline for quantum PDF knowledge extraction with:
- multi-FAISS indexes
- specialized node pipelines
- Neo4j GraphRAG
- DeepSeek-style Streamlit UI (`Black Kite`)
- streaming answer rendering
- cached orchestrator instance
- loading spinner and UI error handling

## Architecture

1. Ingest PDFs (`scripts/ingest.py`)
2. Chunk text (`rag/chunker.py`)
3. Process node pipelines:
   - common terms
   - equations
   - hardware
   - algorithms
   - general corpus
4. Build separate FAISS stores (`scripts/build_vector_db.py`)
5. Build Neo4j concept graph (`scripts/build_graph.py`)
6. Route + retrieve with orchestrator (`agent/orchestrator.py`)
7. Serve via UI (`ui/streamlit_app.py`)

## Windows path to Linux (WSL) mapping

Source PDFs provided at:
`C:\Users\pc\Desktop\Quantum-Computing-Books`

WSL equivalent:
`/mnt/c/Users/pc/Desktop/Quantum-Computing-Books`

If project should live on D drive:

```bash
mkdir -p /mnt/d/quantum-rag-agent
cd /mnt/d/quantum-rag-agent
# copy or clone PDFs into data/raw_pdfs
cp -r /mnt/c/Users/pc/Desktop/Quantum-Computing-Books ./data/raw_pdfs
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run pipeline

```bash
python -m scripts.ingest
python -m scripts.build_vector_db
python -m scripts.build_graph
```

## Run tests

```bash
pytest -q
```

## Launch Black Kite UI

```bash
streamlit run ui/streamlit_app.py
```

## Response contract

All orchestrated responses include:
1. Final Answer
2. Retrieved Context
3. Node Used
4. Related Concepts (Graph)
5. Source PDF

## Notes

- No internet API calls are required.
- Embeddings use `sentence-transformers` with deterministic offline fallback.
- Neo4j credentials are configurable via env vars:
  - `QKRA_NEO4J_URI`
  - `QKRA_NEO4J_USER`
  - `QKRA_NEO4J_PASSWORD`
