"""music-rag source package.

Contains the four building blocks of the RAG pipeline:

    ingestion.py  -> load .txt files, embed them locally, store in ChromaDB
    retriever.py  -> fetch the top-k most relevant chunks for a question
    pipeline.py   -> question + retrieved chunks -> Groq LLM -> answer
    evaluator.py  -> run the 3 misinformation-robustness experiments
"""
