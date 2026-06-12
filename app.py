"""Streamlit UI for the misinformation-robustness RAG demo.

Reads documents from disk and answers questions over them via Groq. Pick a data
source in the sidebar: the hand-written artist files, or a benchmark dataset that
was exported to .txt with `python -m src.export_datasets`.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.ingestion import DocumentIngestor
from src.pipeline import RAGPipeline
from src.retriever import Retriever

# Display name -> (real folder, poisoned folder). The benchmark folders are
# produced by `python -m src.export_datasets`.
SOURCES = {
    "Artist files (data/)": ("data/real", "data/poisoned"),
    "synthetic": ("data/synthetic/real", "data/synthetic/poisoned"),
    "hotpotqa": ("data/hotpotqa/real", "data/hotpotqa/poisoned"),
    "fever": ("data/fever/real", "data/fever/poisoned"),
}

st.set_page_config(page_title="RAG Misinformation Tester", page_icon="🎵")
st.title("🎵 RAG Misinformation Robustness Tester")
st.caption(
    "Ask questions and watch how poisoned documents and verification mode "
    "change the answer."
)


@st.cache_resource(show_spinner=False)
def get_ingestor() -> DocumentIngestor:
    return DocumentIngestor()


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner="Building knowledge base...")
def build_retriever(
    real_dir: str, poisoned_dir: str, include_poisoned: bool, top_k: int
) -> Retriever:
    # Cache key includes the folders + controls, so any change rebuilds the KB.
    ingestor = get_ingestor()
    source_dirs = [real_dir]
    collection = real_dir.replace("/", "_")
    if include_poisoned:
        source_dirs.append(poisoned_dir)
        collection += "_poisoned"
    vectorstore = ingestor.build_vectorstore(source_dirs, collection)
    return Retriever(vectorstore, top_k=top_k)


with st.sidebar:
    st.header("⚙️ Settings")
    source_name = st.selectbox("Data source", list(SOURCES.keys()))
    real_dir, poisoned_dir = SOURCES[source_name]

    include_poisoned = st.checkbox(
        "Include poisoned documents",
        value=True,
        help="Add the fake / poison files to the knowledge base.",
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
    if source_name != "Artist files (data/)":
        st.caption(
            "Benchmark folders come from `python -m src.export_datasets`. "
            f"See `data/{source_name}/index.csv` for the questions to ask."
        )


question = st.text_input(
    "Ask a question:",
    placeholder="e.g. In what year was the band Quantum Echo formed?",
)

if st.button("Get Answer", type="primary") and question.strip():
    try:
        retriever = build_retriever(real_dir, poisoned_dir, include_poisoned, top_k)
        pipeline = get_pipeline()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Retrieving sources and asking the model..."):
        docs = retriever.retrieve(question)
        answer = pipeline.answer(question, docs, verification=verification)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved sources")
    if not docs:
        st.info(
            "No documents retrieved. If you picked a benchmark dataset, run "
            "`python -m src.export_datasets` first."
        )
    for i, doc in enumerate(docs, start=1):
        is_poisoned = doc.metadata.get("is_poisoned")
        source = doc.metadata.get("source", "unknown")
        badge = "☠️ POISONED" if is_poisoned else "✅ real"
        with st.expander(f"Source {i}: {source}  ({badge})"):
            st.write(doc.page_content)
