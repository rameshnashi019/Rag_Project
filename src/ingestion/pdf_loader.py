"""Load PDF documents with LangChain's PyPDFLoader."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdfs(
    sources: str | Path | Iterable[str | Path],
    *,
    recursive: bool = False,
) -> list[Document]:
    """Load pages from one or more PDF files.

    ``sources`` may be a PDF path, a directory, or an iterable of paths.
    Files that cannot be read are logged and skipped so one bad PDF does not
    prevent the remaining files from loading.
    """
    pdf_paths = _find_pdf_paths(sources, recursive=recursive)
    documents: list[Document] = []

    if not pdf_paths:
        logger.warning("No PDF files found in %s", sources)
        return documents

    for pdf_path in pdf_paths:
        try:
            logger.info("Loading PDF: %s", pdf_path)
            loaded_documents = PyPDFLoader(str(pdf_path)).load()
            documents.extend(loaded_documents)
            logger.info("Loaded %d page(s) from %s", len(loaded_documents), pdf_path)
        except Exception:
            logger.exception("Failed to load PDF: %s", pdf_path)

    logger.info("Loaded %d page(s) from %d PDF file(s)", len(documents), len(pdf_paths))
    return documents


def _find_pdf_paths(
    sources: str | Path | Iterable[str | Path],
    *,
    recursive: bool,
) -> list[Path]:
    """Normalize input sources into sorted, unique PDF paths."""
    if isinstance(sources, (str, Path)):
        source_paths: Iterable[str | Path] = [sources]
    else:
        source_paths = sources

    pdf_paths: set[Path] = set()
    for source in source_paths:
        path = Path(source)
        if path.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            pdf_paths.update(candidate for candidate in path.glob(pattern) if candidate.is_file())
        elif path.is_file() and path.suffix.lower() == ".pdf":
            pdf_paths.add(path)
        else:
            logger.warning("Skipping missing or non-PDF source: %s", path)

    return sorted(pdf_paths, key=lambda path: str(path).lower())