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
.kite-title { font-size: 28px; font-weight: 700; color: #e6e9ef; margin-bottom: 4px; }
.kite-sub { color:#99a2ad; margin-bottom: 16px; }
.section { background:#12161c; border:1px solid #242a33; border-radius:10px; padding:12px; margin-top:10px; }
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


@st.cache_resource
def get_orchestrator() -> AgentOrchestrator:
    """Create one cached orchestrator instance."""
    return AgentOrchestrator()


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

st.markdown("<div class='kite-title'>Black Kite • Quantum Knowledge Retrieval Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='kite-sub'>DeepSeek-style local chat over quantum PDF corpus</div>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about equations, hardware, algorithms, or concepts...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            orchestrator = get_orchestrator()
            with st.spinner("Black Kite is retrieving vector + graph context..."):
                result = orchestrator.run(
                    query,
                    node_override=node_choice,
                    vector_override=vector_choice,
                    history=st.session_state.history,
                )

            st.markdown("#### Answer")
            streamed_text = st.write_stream(orchestrator.stream_text(result.answer))

            st.markdown("#### Retrieved Context")
            st.markdown(f"<div class='section'>{result.retrieved_context or 'No context found.'}</div>", unsafe_allow_html=True)

            st.markdown("#### Graph Concepts")
            graph_text = ", ".join(result.related_concepts) if result.related_concepts else "None"
            st.markdown(f"<div class='section'>{graph_text}</div>", unsafe_allow_html=True)

            st.markdown("#### Node / Source")
            st.markdown(f"**Node Used:** {result.node_used}  ")
            st.markdown(f"**Source PDF:** {result.source_pdf}")

            stored = (
                f"**Answer**\n{streamed_text}\n\n"
                f"**Retrieved Context**\n{result.retrieved_context}\n\n"
                f"**Graph Concepts**\n{graph_text}\n\n"
                f"**Node Used:** {result.node_used}\n\n"
                f"**Source PDF:** {result.source_pdf}"
            )
            st.session_state.history.append({"role": "assistant", "content": stored})
        except Exception as exc:
            error_message = f"Black Kite error: {exc}"
            st.error(error_message)
            st.session_state.history.append({"role": "assistant", "content": error_message})
