"""ingestion.py

Loads plain-text documents from disk, embeds them with a *local* embedding
model (no API calls / no cost), and stores the resulting vectors in a ChromaDB
collection so they can be retrieved later.

Design notes for students
--------------------------
* Embeddings are produced by `sentence-transformers/all-MiniLM-L6-v2`, which
  runs entirely on your machine. The first run downloads the model (~90 MB);
  after that it is cached and works offline.
* Each "experiment" uses its own ChromaDB collection so the knowledge bases
  stay isolated (e.g. "real only" vs "real + poisoned"). That is why
  `build_vectorstore` takes a `collection_name` and a list of source folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Name of the local sentence-transformers model used for embeddings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Folder where ChromaDB persists its data on disk.
DEFAULT_PERSIST_DIR = "chroma_db"


class DocumentIngestor:
    """Loads `.txt` files, embeds them locally, and writes them to ChromaDB.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB stores the vector database on disk.
    chunk_size / chunk_overlap:
        Controls how long documents are split into smaller, overlapping chunks
        before embedding. Smaller chunks give more precise retrieval.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.persist_dir = persist_dir

        # The embedding model runs locally — this is the "no paid API" part.
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # Splits long text into overlapping chunks for finer-grained retrieval.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------ #
    # Loading raw text files into LangChain Document objects
    # ------------------------------------------------------------------ #
    def _load_documents(self, source_dirs: Iterable[str]) -> List[Document]:
        """Read every `.txt` file in the given folders into `Document`s.

        Each document is tagged with metadata describing where it came from,
        including whether it lives in a folder named "poisoned". That tag is
        handy in the UI so the user can *see* when a fake source was retrieved.
        """
        documents: List[Document] = []

        for source_dir in source_dirs:
            folder = Path(source_dir)
            if not folder.exists():
                # Skip silently — the user may not have added data yet.
                continue

            for txt_path in sorted(folder.glob("*.txt")):
                text = txt_path.read_text(encoding="utf-8").strip()
                if not text:
                    continue  # ignore empty placeholder files

                is_poisoned = "poisoned" in txt_path.parts
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": txt_path.name,
                            "source_path": str(txt_path),
                            "is_poisoned": is_poisoned,
                        },
                    )
                )

        return documents

    # ------------------------------------------------------------------ #
    # Building / refreshing a vector store collection
    # ------------------------------------------------------------------ #
    def build_vectorstore(
        self,
        source_dirs: Iterable[str],
        collection_name: str,
    ) -> Chroma:
        """Embed all docs in `source_dirs` and persist them under `collection_name`.

        Any existing collection with the same name is wiped first so each call
        produces a clean, reproducible knowledge base (important for the
        experiments, which must not leak documents between runs).
        """
        raw_docs = self._load_documents(source_dirs)
        chunks = self.splitter.split_documents(raw_docs)

        # Start from an empty collection so reruns are deterministic.
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        # Drop anything left over from a previous run, then add fresh chunks.
        try:
            vectorstore.delete_collection()
        except Exception:
            pass  # collection may not exist yet on the very first run

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_dir,
        )

        print(
            f"[ingestion] collection '{collection_name}': "
            f"{len(raw_docs)} files -> {len(chunks)} chunks embedded."
        )
        return vectorstore

    def build_vectorstore_from_documents(
        self,
        documents: List[Document],
        collection_name: str = "in_memory",
    ) -> Chroma:
        """Build an **ephemeral, in-memory** vector store from `Document`s.

        Unlike :meth:`build_vectorstore`, this does not read folders or persist
        to disk — it embeds the documents you pass in and keeps the collection
        in memory only. This is used by the evaluator to build a fresh, tiny
        knowledge base *per question* (e.g. one HotpotQA item's passages plus an
        optional poison doc), which keeps the benchmark runs fast and isolated.

        Passing `persist_directory=None` makes Chroma use an in-memory client,
        so nothing is written to disk and collections never collide between
        questions.
        """
        chunks = self.splitter.split_documents(documents)
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=None,  # in-memory only
        )

    def load_vectorstore(self, collection_name: str) -> Chroma:
        """Open an already-built collection without re-embedding anything."""
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )


if __name__ == "__main__":
    # Quick manual test: build a knowledge base from the real + poisoned folders.
    ingestor = DocumentIngestor()
    ingestor.build_vectorstore(
        source_dirs=["data/real", "data/poisoned"],
        collection_name="manual_test",
    )
