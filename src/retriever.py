"""Top-k retrieval over a ChromaDB collection."""

from __future__ import annotations

from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class Retriever:
    def __init__(self, vectorstore: Chroma, top_k: int = 4) -> None:
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Document]:
        return self.vectorstore.similarity_search(question, k=self.top_k)

    def retrieve_with_scores(self, question: str):
        """Like retrieve, but also returns the similarity score per chunk."""
        return self.vectorstore.similarity_search_with_score(question, k=self.top_k)
