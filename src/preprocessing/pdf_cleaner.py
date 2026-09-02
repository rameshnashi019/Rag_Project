"""Clean text extracted from PDFs before chunking or indexing."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def clean_documents(
    documents: Iterable[Document],
    *,
    header_patterns: Sequence[str] = (),
    footer_patterns: Sequence[str] = (),
    watermark_patterns: Sequence[str] = (),
    text_patterns: Sequence[tuple[str, str]] = (),
) -> list[Document]:
    """Clean extracted PDF documents while preserving their metadata.

    Patterns are regular expressions matched against individual lines. Header,
    footer, and watermark patterns remove matching lines. ``text_patterns``
    accepts ``(pattern, replacement)`` pairs for other targeted substitutions.
    Invalid regular expressions are logged and ignored.
    """
    compiled_header = _compile_patterns(header_patterns, "header")
    compiled_footer = _compile_patterns(footer_patterns, "footer")
    compiled_watermark = _compile_patterns(watermark_patterns, "watermark")
    compiled_text = _compile_replacements(text_patterns)

    cleaned_documents: list[Document] = []
    for document in documents:
        try:
            cleaned_text = _clean_text(
                document.page_content,
                header_patterns=compiled_header,
                footer_patterns=compiled_footer,
                watermark_patterns=compiled_watermark,
                text_patterns=compiled_text,
            )
            cleaned_documents.append(
                Document(page_content=cleaned_text, metadata=dict(document.metadata))
            )
            logger.info(
                "Cleaned document source=%s characters=%d -> %d",
                document.metadata.get("source", "unknown"),
                len(document.page_content),
                len(cleaned_text),
            )
        except Exception:
            logger.exception(
                "Failed to clean document source=%s",
                document.metadata.get("source", "unknown"),
            )

    logger.info("Cleaned %d document(s)", len(cleaned_documents))
    return cleaned_documents


def _clean_text(
    text: str,
    *,
    header_patterns: Sequence[re.Pattern[str]],
    footer_patterns: Sequence[re.Pattern[str]],
    watermark_patterns: Sequence[re.Pattern[str]],
    text_patterns: Sequence[tuple[re.Pattern[str], str]],
) -> str:
    normalized_text = unicodedata.normalize("NFKC", text)
    normalized_text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", normalized_text)
    lines = normalized_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: list[str] = []

    last_line_index = len(lines) - 1
    for line_index, line in enumerate(lines):
        normalized_line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if not normalized_line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if line_index in {0, last_line_index} and re.fullmatch(r"\d+", normalized_line):
            continue
        if any(pattern.search(normalized_line) for pattern in (
            *header_patterns,
            *footer_patterns,
            *watermark_patterns,
        )):
            continue
        cleaned_lines.append(normalized_line)

    cleaned_text = "\n".join(cleaned_lines).strip()
    for pattern, replacement in text_patterns:
        cleaned_text = pattern.sub(replacement, cleaned_text)

    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    return re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()


def _compile_patterns(patterns: Sequence[str], pattern_type: str) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            logger.exception("Ignoring invalid %s regex pattern: %s", pattern_type, pattern)
    return compiled


def _compile_replacements(
    patterns: Sequence[tuple[str, str]],
) -> list[tuple[re.Pattern[str], str]]:
    compiled: list[tuple[re.Pattern[str], str]] = []
    for pattern, replacement in patterns:
        try:
            compiled.append((re.compile(pattern, re.IGNORECASE), replacement))
        except re.error:
            logger.exception("Ignoring invalid text regex pattern: %s", pattern)
    return compiled