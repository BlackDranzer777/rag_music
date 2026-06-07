"""retriever.py

Thin wrapper around a ChromaDB collection that, given a natural-language
question, returns the top-k most relevant document chunks.

Keeping retrieval in its own class means the pipeline does not need to know
*how* documents are fetched — it just asks for "the relevant context".
"""

from __future__ import annotations

from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class Retriever:
    """Returns the most relevant chunks for a question via vector similarity.

    Parameters
    ----------
    vectorstore:
        A ChromaDB collection produced by `DocumentIngestor`.
    top_k:
        How many chunks to return for each query.
    """

    def __init__(self, vectorstore: Chroma, top_k: int = 4) -> None:
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Document]:
        """Return the `top_k` chunks most similar to `question`."""
        return self.vectorstore.similarity_search(question, k=self.top_k)

    def retrieve_with_scores(self, question: str):
        """Like `retrieve`, but also returns the similarity score per chunk.

        Useful in the UI / debugging to see *how* confident retrieval was.
        Returns a list of (Document, score) tuples.
        """
        return self.vectorstore.similarity_search_with_score(question, k=self.top_k)
