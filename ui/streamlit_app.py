"""Black Kite UI: DeepSeek-like offline interface."""

from __future__ import annotations

import streamlit as st

from agent.orchestrator import AgentOrchestrator

st.set_page_config(page_title="Black Kite • QKRA", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"] { background-color: #0b0d10; color: #d7dbe0; }
[data-testid="stSidebar"] { background-color: #11151b; }
.chat-shell { background:#12161c; border:1px solid #242a33; padding:18px; border-radius:12px; }
.kite-title { font-size: 28px; font-weight: 700; color: #e6e9ef; margin-bottom: 4px; }
.kite-sub { color:#99a2ad; margin-bottom: 16px; }
.small { color:#98a1ab; font-size:13px; }
</style>
""",
    unsafe_allow_html=True,
)

LOGO_SVG = """
<svg width="36" height="36" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 26 L24 4 L44 26 L24 20 Z" fill="#7f8ea3"/>
  <path d="M24 20 L34 44 L24 35 L14 44 Z" fill="#5d6877"/>
</svg>
"""

with st.sidebar:
    st.markdown(f"{LOGO_SVG}", unsafe_allow_html=True)
    st.markdown("### Black Kite")
    node_choice = st.selectbox(
        "Node Selection",
        ["auto", "general", "common_terms", "equation", "hardware", "algorithm"],
        index=0,
    )
    vector_choice = st.selectbox(
        "Vector DB Selection",
        ["auto", "general", "common_terms", "equation", "hardware", "algorithm"],
        index=0,
    )
    st.markdown("---")
    st.markdown("Offline-first Agentic RAG for quantum knowledge.")

st.markdown("<div class='kite-title'>Black Kite • Quantum Knowledge Retrieval Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='kite-sub'>DeepSeek-style local chat over quantum PDF corpus</div>", unsafe_allow_html=True)

query = st.chat_input("Ask about equations, hardware, algorithms, or quantum concepts...")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    orchestrator = AgentOrchestrator()
    result = orchestrator.run(query, node_override=node_choice, vector_override=vector_choice)
    response_text = (
        f"**Answer**\n{result.answer}\n\n"
        f"**Retrieved Context**\n{result.retrieved_context}\n\n"
        f"**Node Used** {result.node_used}\n\n"
        f"**Related Concepts** {', '.join(result.related_concepts) if result.related_concepts else 'None'}\n\n"
        f"**Source PDF** {result.source_pdf}"
    )
    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.history.append({"role": "assistant", "content": response_text})
