"""DeepSeek-like minimal chat UI for QKRA."""

import streamlit as st

from agent.orchestrator import orchestrate

st.set_page_config(page_title="Quantum RAG Assistant", layout="wide")
st.title("Quantum Knowledge Retrieval Agent (QKRA)")

query = st.text_input("Ask about quantum computing concepts, equations, hardware, or algorithms")
if query:
    st.text(orchestrate(query))
