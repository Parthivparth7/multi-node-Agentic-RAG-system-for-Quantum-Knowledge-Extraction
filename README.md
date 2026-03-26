# Quantum Knowledge Retrieval Agent (QKRA)

Offline-first, multi-node Agentic RAG system for Quantum Computing books/PDFs.

## 1) Project Theme (What this repo builds)

This project is a **Quantum Knowledge Operating System (QKOS)** style stack:
- Ingest 50+ quantum PDFs.
- Build **multiple FAISS indexes** for specialized retrieval.
- Run **node-wise extraction agents** (terms/equations/hardware/algorithms/graph).
- Route user query via orchestrator.
- Return response in strict format:
  1. Answer
  2. Supporting Context
  3. Related Concepts
  4. Source (PDF name)

## 2) Current Folder Layout

```text
.
├── agent/
├── config/
├── data/
│   ├── raw_pdfs/
│   └── processed_chunks/
├── embeddings/
├── graph_db/
├── nodes/
├── rag/
├── scripts/
├── tests/
├── ui/
└── vector_db/
```

## 3) PDF Location Mapping (Windows path -> Linux path)

You shared Windows path:

```text
C:\Users\pc\Desktop\Quantum-Computing-Books
```

If you are using WSL/Linux shell, this typically maps to:

```bash
/mnt/c/Users/pc/Desktop/Quantum-Computing-Books
```

You said you want project on **D drive**. Example setup:

```bash
mkdir -p /mnt/d/quantum-rag-agent
cp -r /mnt/c/Users/pc/Desktop/Quantum-Computing-Books /mnt/d/quantum-rag-agent/data/raw_pdfs
```

Or clone directly on D:

```bash
cd /mnt/d
git clone <your-repo-url> quantum-rag-agent
cd quantum-rag-agent
git clone https://github.com/manjunath5496/Quantum-Computing-Books.git data/raw_pdfs
```

## 4) Linux Commands to Verify Project Works

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Check 1: Ingest PDFs

```bash
python -m scripts.ingest
```

Expected: `data/processed_chunks/chunks.jsonl` created.

### Check 2: Build FAISS indexes

```bash
python -m scripts.build_vector_db
```

Expected indexes under:
- `vector_db/faiss_common_terms/`
- `vector_db/faiss_equations/`
- `vector_db/faiss_hardware/`
- `vector_db/faiss_algorithms/`
- `vector_db/faiss_general/`

### Check 3: Run tests

```bash
pytest -q
```

### Check 4: Start UI

```bash
streamlit run ui/streamlit_app.py
```

## 5) Node-wise Codex Agent Prompts (each node separate agent)

Use one prompt per Codex agent session.

### Agent A — Common Terms Node

```text
Implement/upgrade nodes/node_common_terms.py.
Goal: domain-specific term extraction from quantum chunks.
Requirements:
- keep function extract_common_terms(chunks, top_n=100)
- support TF-IDF baseline
- add optional KeyBERT mode (offline model only)
- output list[str] sorted by score desc
- add unit tests in tests/test_common_terms.py
- do not modify other nodes
```

### Agent B — Equation Node

```text
Implement/upgrade nodes/node_equations.py.
Goal: robust equation extraction.
Requirements:
- detect inline LaTeX ($...$), block (\[...\]), equation env
- return cleaned equations and deduplicate
- add equation metadata (source, page, chunk_id) helper
- add tests in tests/test_equations.py
```

### Agent C — Hardware Node

```text
Implement/upgrade nodes/node_hardware.py.
Goal: detect hardware families in chunk text.
Requirements:
- classify superconducting, trapped_ions, photonic
- keyword + pattern hybrid approach
- return dict with booleans and confidence score per class
- tests in tests/test_hardware.py
```

### Agent D — Algorithm Node

```text
Implement/upgrade nodes/node_algorithms.py.
Goal: detect algorithm mentions and explanation chunks.
Requirements:
- detect Shor, Grover, VQE, QAOA
- return structured result {algorithm: {detected, evidence}}
- add extractor for nearest explanation sentence
- tests in tests/test_algorithms.py
```

### Agent E — Graph Node (Neo4j)

```text
Implement/upgrade nodes/node_graph.py.
Goal: production-ready GraphRAG graph layer.
Requirements:
- schema: (Concept)-[:RELATED]->(Concept)
- add relation weights and source fields
- add upsert batch API
- add query for top related concepts by weight
- tests with mock driver in tests/test_graph_node.py
```

### Agent F — Orchestrator + Router

```text
Implement/upgrade agent/planner.py and agent/orchestrator.py.
Goal: robust node routing + response formatting.
Requirements:
- query intent routing for equation/hardware/algorithm/graph/general
- call vector retrieval + optional graph expansion
- strict output format: Answer, Supporting Context, Related Concepts, Source
- add tests in tests/test_orchestrator.py
```

## 6) GraphRAG Upgrade Plan (Very Powerful)

1. **Entity + relation extraction layer**
   - entities: Concept, Algorithm, Hardware, Equation, Paper
   - relations: RELATED, USES, IMPLEMENTS, DERIVED_FROM, LIMITS

2. **Hybrid retrieval**
   - Stage A: FAISS top-k chunks
   - Stage B: Neo4j neighborhood expansion from extracted entities
   - Stage C: rerank merged context (cross-encoder optional offline)

3. **Multi-hop reasoning**
   - 2-hop traversal for professor-level queries
   - path explanation in final response (why concept A links to B)

4. **Confidence & grounding**
   - show chunk IDs and PDF/page references
   - drop claims without support

5. **Graph maintenance jobs**
   - periodic deduplication of concept nodes
   - edge weight refresh using co-occurrence + citation frequency

## 7) Production Notes

- Keep all models local/offline (sentence-transformers + local LLM via Ollama/llama.cpp).
- Avoid hardcoded paths; use `config/settings.py`.
- Make each module independently testable.

---

## Quick Start (TL;DR)

```bash
pip install -r requirements.txt
python -m scripts.ingest
python -m scripts.build_vector_db
pytest -q
streamlit run ui/streamlit_app.py
```
