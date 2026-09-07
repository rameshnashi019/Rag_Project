"""End-to-end retrieval-augmented generation pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from chunking import chunk_documents
from generation import generate_answer
from ingestion import load_documents
from preprocessing import clean_documents
from reterieval import retrieve
from vectordb import ChromaStore, create_vector_store

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Connect document loading, cleaning, chunking, storage, retrieval, and generation."""

    def __init__(
        self,
        *,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = "pdf_documents",
    ) -> None:
        self.store: ChromaStore = create_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def index(
        self,
        sources: str | Path,
        *,
        recursive: bool = False,
        reset: bool = False,
        header_patterns: tuple[str, ...] = (),
        footer_patterns: tuple[str, ...] = (),
        watermark_patterns: tuple[str, ...] = (),
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """Load, clean, chunk, and persist supported documents."""
        if reset:
            self.store.clear()
        load_dotenv()
        documents = load_documents(sources, recursive=recursive)
        cleaned_documents = clean_documents(
            documents,
            header_patterns=header_patterns,
            footer_patterns=footer_patterns,
            watermark_patterns=watermark_patterns,
        )
        chunks = chunk_documents(
            cleaned_documents,
            chunk_size=(
                chunk_size
                if chunk_size is not None
                else _get_chunk_setting("CHUNK_SIZE", 1200)
            ),
            chunk_overlap=(
                chunk_overlap
                if chunk_overlap is not None
                else _get_chunk_setting("CHUNK_OVERLAP", 200)
            ),
        )
        self.store.add_documents(chunks)
        logger.info("Indexed %d chunk(s) from %d page(s)", len(chunks), len(documents))
        return len(chunks)

    def ask(self, question: str, *, k: int = 4) -> str:
        """Retrieve relevant chunks and generate a grounded answer."""
        answer, _ = self.ask_with_sources(question, k=k)
        return answer

    def ask_with_sources(self, question: str, *, k: int = 4) -> tuple[str, list[dict[str, object]]]:
        """Generate a grounded answer and return its source metadata."""
        retrieval_result = retrieve(question, self.store, k=k, decompose=True, use_mmr=True)
        answer = generate_answer(question, retrieval_result)
        sources = [
            {
                "source": document.metadata.get("source", "unknown"),
                "page": document.metadata.get("page"),
                "chunk_index": document.metadata.get("chunk_index"),
            }
            for document in retrieval_result.documents
        ]
        return answer, sources


def _get_chunk_setting(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error