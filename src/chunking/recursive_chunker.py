"""Split cleaned documents into recursive, metadata-rich chunks."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    separators: Sequence[str] | None = None,
) -> list[Document]:
    """Split documents recursively while preserving and enriching metadata.

    Larger semantic boundaries are preferred first: paragraphs, lines, words,
    and finally characters. Each output chunk receives ``chunk_index`` and
    ``chunk_count`` metadata in addition to the source document metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators is not None else ["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks: list[Document] = []
    processed_documents = 0

    for document in documents:
        try:
            document_chunks = splitter.split_documents([document])
            chunk_count = len(document_chunks)
            for chunk_index, chunk in enumerate(document_chunks):
                chunk.metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "chunk_size": len(chunk.page_content),
                    }
                )
            chunks.extend(document_chunks)
            processed_documents += 1
            logger.info(
                "Created %d chunk(s) from source=%s page=%s",
                chunk_count,
                document.metadata.get("source", "unknown"),
                document.metadata.get("page", "unknown"),
            )
        except Exception:
            logger.exception(
                "Failed to chunk document source=%s page=%s",
                document.metadata.get("source", "unknown"),
                document.metadata.get("page", "unknown"),
            )

    logger.info(
        "Created %d chunk(s) from %d document(s)",
        len(chunks),
        processed_documents,
    )
    return chunks