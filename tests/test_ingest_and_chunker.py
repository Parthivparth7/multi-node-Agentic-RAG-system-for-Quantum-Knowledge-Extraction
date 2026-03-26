from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from scripts.ingest import load_pdf, process_all, save_chunks


def test_pdf_ingestion_pipeline(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quantum computing uses qubits and Grover algorithm.")
    doc.save(pdf_path)
    doc.close()

    records = load_pdf(pdf_path)
    assert records and records[0]["source"] == "sample.pdf"

    out = process_all(tmp_path)
    assert out and "chunk" in out[0]

    out_file = tmp_path / "chunks.jsonl"
    save_chunks(out, out_file)
    assert out_file.exists()
