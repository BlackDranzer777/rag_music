"""Load text documents, embed them locally, and store them in ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "chroma_db"


class DocumentIngestor:
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _load_documents(self, source_dirs: Iterable[str]) -> List[Document]:
        documents: List[Document] = []

        for source_dir in source_dirs:
            folder = Path(source_dir)
            if not folder.exists():
                continue

            for txt_path in sorted(folder.glob("*.txt")):
                text = txt_path.read_text(encoding="utf-8").strip()
                if not text:
                    continue

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

    def build_vectorstore(
        self,
        source_dirs: Iterable[str],
        collection_name: str,
    ) -> Chroma:
        """Embed all docs in source_dirs and persist them under collection_name."""
        raw_docs = self._load_documents(source_dirs)
        chunks = self.splitter.split_documents(raw_docs)

        # Wipe any existing collection so reruns stay deterministic.
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        try:
            vectorstore.delete_collection()
        except Exception:
            pass

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
        """Build an in-memory vector store from documents (nothing written to disk)."""
        chunks = self.splitter.split_documents(documents)
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=None,
        )

    def load_vectorstore(self, collection_name: str) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )


if __name__ == "__main__":
    ingestor = DocumentIngestor()
    ingestor.build_vectorstore(
        source_dirs=["data/real", "data/poisoned"],
        collection_name="manual_test",
    )
