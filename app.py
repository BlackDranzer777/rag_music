"""app.py

Streamlit UI for the misinformation-robustness RAG demo.

Features
--------
* Sidebar controls to choose which documents are in the knowledge base
  (real only, or real + poisoned) and to (re)build the vector store.
* A toggle for "verification mode" (the skeptical / contradiction-checking
  prompt).
* A text box to ask a question; the answer is shown along with the retrieved
  source chunks, clearly flagging any POISONED source that was retrieved.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.ingestion import DocumentIngestor
from src.pipeline import RAGPipeline
from src.retriever import Retriever

REAL_DIR = "data/real"
POISONED_DIR = "data/poisoned"

st.set_page_config(page_title="RAG Misinformation Tester", page_icon="🎵")
st.title("🎵 RAG Misinformation Robustness Tester")
st.caption(
    "Ask questions about music artists and watch how poisoned documents and "
    "verification mode change the answer."
)


# --------------------------------------------------------------------------- #
# Cached resources — built once and reused across reruns for speed.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_ingestor() -> DocumentIngestor:
    return DocumentIngestor()


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner="Building knowledge base...")
def build_retriever(include_poisoned: bool, top_k: int) -> Retriever:
    """Build (or rebuild) the vector store and wrap it in a Retriever.

    The cache key includes `include_poisoned` and `top_k`, so flipping either
    control transparently rebuilds the right knowledge base.
    """
    ingestor = get_ingestor()
    source_dirs = [REAL_DIR]
    collection = "ui_real_only"
    if include_poisoned:
        source_dirs.append(POISONED_DIR)
        collection = "ui_real_poisoned"
    vectorstore = ingestor.build_vectorstore(source_dirs, collection)
    return Retriever(vectorstore, top_k=top_k)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    include_poisoned = st.checkbox(
        "Include poisoned documents",
        value=True,
        help="Add the fake docs in data/poisoned to the knowledge base.",
    )
    verification = st.toggle(
        "Verification mode",
        value=False,
        help="Instruct the model to cross-check sources for contradictions "
        "before answering.",
    )
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)

    st.divider()
    st.markdown(
        "**Knowledge base:** "
        + ("Real + Poisoned" if include_poisoned else "Real only")
    )


# --------------------------------------------------------------------------- #
# Main interaction
# --------------------------------------------------------------------------- #
question = st.text_input(
    "Ask a question about a music artist:",
    placeholder="e.g. In what year was the band Quantum Echo formed?",
)

if st.button("Get Answer", type="primary") and question.strip():
    try:
        retriever = build_retriever(include_poisoned, top_k)
        pipeline = get_pipeline()
    except ValueError as exc:
        # Most common cause: missing GROQ_API_KEY in .env
        st.error(str(exc))
        st.stop()

    with st.spinner("Retrieving sources and asking the model..."):
        docs = retriever.retrieve(question)
        answer = pipeline.answer(question, docs, verification=verification)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved sources")
    if not docs:
        st.info("No documents retrieved. Have you added files to data/ ?")
    for i, doc in enumerate(docs, start=1):
        is_poisoned = doc.metadata.get("is_poisoned")
        source = doc.metadata.get("source", "unknown")
        badge = "☠️ POISONED" if is_poisoned else "✅ real"
        with st.expander(f"Source {i}: {source}  ({badge})"):
            st.write(doc.page_content)
