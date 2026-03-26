"""Black Kite UI: DeepSeek-like offline interface."""

from __future__ import annotations

import logging

import streamlit as st

from agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Black Kite • QKRA", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"] { background-color: #0b0d10; color: #d7dbe0; }
[data-testid="stSidebar"] { background-color: #11151b; }
.kite-title { font-size: 28px; font-weight: 700; color: #e6e9ef; margin-bottom: 4px; }
.kite-sub { color:#99a2ad; margin-bottom: 16px; }
.section { background:#12161c; border:1px solid #242a33; border-radius:10px; padding:12px; margin-top:10px; }
.msg-divider { margin-top: 8px; margin-bottom: 14px; }
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
    show_debug = st.checkbox("Show Debug Info", value=False)

st.markdown("<div class='kite-title'>Black Kite • Quantum Knowledge Retrieval Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='kite-sub'>DeepSeek-style local chat over quantum PDF corpus</div>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.markdown("<div class='msg-divider'></div>", unsafe_allow_html=True)

query = st.chat_input("Ask about equations, hardware, algorithms, or concepts...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        st.markdown("<div class='msg-divider'></div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        try:
            orchestrator = get_orchestrator()
            with st.spinner("Black Kite is retrieving vector + graph context..."):
                # Use true stream interface when available; fallback kept for compatibility.
                if hasattr(orchestrator, "stream"):
                    stream_iter, result = orchestrator.stream(
                        query,
                        node_override=node_choice,
                        vector_override=vector_choice,
                        history=st.session_state.history,
                    )
                else:
                    result = orchestrator.run(
                        query,
                        node_override=node_choice,
                        vector_override=vector_choice,
                        history=st.session_state.history,
                    )
                    stream_iter = orchestrator.stream_text(result.answer)

            st.markdown("#### 🧠 Answer")
            streamed_answer = st.write_stream(stream_iter)
            st.info(f"⚙️ Node: {result.node_used}")

            if show_debug:
                with st.expander("Retrieved Context", expanded=False):
                    st.markdown(f"<div class='section'>{result.retrieved_context or 'No context found.'}</div>", unsafe_allow_html=True)
                with st.expander("🔗 Graph Concepts", expanded=False):
                    graph_text = ", ".join(result.related_concepts) if result.related_concepts else "None"
                    st.markdown(f"<div class='section'>{graph_text}</div>", unsafe_allow_html=True)
                with st.expander("📄 Source", expanded=False):
                    st.markdown(f"<div class='section'>{result.source_pdf}</div>", unsafe_allow_html=True)

            # Keep history clean: only raw answer text.
            st.session_state.history.append({"role": "assistant", "content": streamed_answer})
            st.markdown("<div class='msg-divider'></div>", unsafe_allow_html=True)
        except Exception as exc:
            logger.exception("Black Kite UI error")
            st.error("Sorry, something went wrong while generating the response. Please try again.")
            st.session_state.history.append(
                {"role": "assistant", "content": "Sorry, something went wrong while generating the response. Please try again."}
            )
