"""Persistent Chroma vector store for embedded documents."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from enbeddings import create_embeddings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Provide document storage and similarity search with Chroma."""

    def __init__(self, vector_store: Chroma) -> None:
        self.vector_store = vector_store

    def add_documents(self, documents: Iterable[Document]) -> list[str]:
        """Embed and persist documents in Chroma."""
        document_list = list(documents)
        if not document_list:
            logger.warning("No documents supplied for Chroma ingestion")
            return []

        try:
            ids = self.vector_store.add_documents(document_list)
            logger.info("Added %d document chunk(s) to Chroma", len(document_list))
            return ids
        except Exception:
            logger.exception("Failed to add documents to Chroma")
            return []

    def search(self, query: str, *, k: int = 4) -> list[Document]:
        """Return the most similar documents for a query."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info("Found %d result(s) for vector search", len(results))
            return results
        except Exception:
            logger.exception("Failed to search Chroma")
            return []

    def search_mmr(self, query: str, *, k: int = 4, fetch_k: int = 20) -> list[Document]:
        """Return diverse relevant documents using maximal marginal relevance."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0 or fetch_k < k:
            raise ValueError("fetch_k must be at least k, and both must be greater than zero")

        try:
            results = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
            )
            logger.info("Found %d diverse result(s) for MMR search", len(results))
            return results
        except Exception:
            logger.exception("Failed to run MMR search")
            return []

    def clear(self) -> None:
        """Delete all documents from the current Chroma collection."""
        try:
            collection_ids = self.vector_store._collection.get(include=[]).get("ids", [])
            if collection_ids:
                self.vector_store._collection.delete(ids=collection_ids)
            logger.info("Cleared %d document(s) from Chroma collection", len(collection_ids))
        except Exception:
            logger.exception("Failed to clear Chroma collection")
            raise


def create_vector_store(
    *,
    persist_directory: str | Path = "chroma_db",
    collection_name: str = "pdf_documents",
    embeddings: Embeddings | None = None,
    documents: Iterable[Document] | None = None,
    distance_metric: str = "cosine",
) -> ChromaStore:
    """Create a persistent Chroma store and optionally ingest documents."""
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")
    if distance_metric not in {"cosine", "l2", "ip"}:
        raise ValueError("distance_metric must be 'cosine', 'l2', or 'ip'")

    try:
        directory = Path(persist_directory)
        directory.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings or create_embeddings(),
            persist_directory=str(directory),
            collection_metadata={"hnsw:space": distance_metric},
        )
        store = ChromaStore(vector_store)
        if documents is not None:
            store.add_documents(documents)
        logger.info("Initialized Chroma collection '%s' at %s", collection_name, directory)
        return store
    except Exception:
        logger.exception("Failed to initialize Chroma collection '%s'", collection_name)
        raise