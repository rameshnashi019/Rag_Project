"""Retrieve relevant chunks from the Chroma vector store."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from vectordb import ChromaStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Hold retrieved documents and generation context."""

    documents: list[Document]
    context: str


def retrieve(
    query: str,
    store: ChromaStore,
    *,
    k: int = 4,
    decompose: bool = False,
    query_model: BaseChatModel | None = None,
    use_mmr: bool = True,
) -> RetrievalResult:
    """Retrieve relevant chunks using decomposition, fusion, and MMR."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    try:
        queries = _decompose_query(query, query_model) if decompose else [query]
        documents = _retrieve_queries(store, queries, k=k, use_mmr=use_mmr)
        context = _format_context(documents)
        logger.info("Retrieved %d chunk(s) for user query", len(documents))
        return RetrievalResult(documents=documents, context=context)
    except Exception:
        logger.exception("Failed to retrieve chunks for user query")
        return RetrievalResult(documents=[], context="")


def _retrieve_queries(
    store: ChromaStore,
    queries: list[str],
    *,
    k: int,
    use_mmr: bool,
) -> list[Document]:
    ranked_documents: dict[str, tuple[float, Document]] = {}
    for query_index, query in enumerate(queries):
        documents = (
            store.search_mmr(query, k=k, fetch_k=max(k * 4, 10))
            if use_mmr
            else store.search(query, k=k)
        )
        for rank, document in enumerate(documents, start=1):
            key = _document_key(document)
            score = 1 / (60 + rank) + 1 / (60 + query_index + rank)
            if key in ranked_documents:
                score += ranked_documents[key][0]
            ranked_documents[key] = (score, document)

    return [document for _, document in sorted(
        ranked_documents.values(), key=lambda item: item[0], reverse=True
    )[:k]]


def _document_key(document: Document) -> str:
    metadata = document.metadata
    return "|".join(
        str(metadata.get(field, ""))
        for field in ("source", "page", "start_index", "chunk_index")
    ) + f"|{document.page_content}"


def _decompose_query(query: str, query_model: BaseChatModel | None) -> list[str]:
    model = query_model or _create_query_model()
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Break the user question into up to three independent search queries. "
                    "Return only one query per line, with no numbering. "
                    "Keep the original question if it is already focused."
                )
            ),
            HumanMessage(content=query),
        ]
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    queries = [line.strip(" -\t") for line in content.splitlines() if line.strip()]
    return list(dict.fromkeys([query, *queries[:3]]))


def _create_query_model() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_CHAT_MODEL")
    if not api_key or not model:
        raise ValueError("OPENAI_API_KEY and OPENAI_CHAT_MODEL must be set in .env")
    return ChatOpenAI(model=model, temperature=0)


def _format_context(documents: list[Document]) -> str:
    context_parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        location = f"source={source}"
        if page is not None:
            location += f", page={page}"
        context_parts.append(f"[Chunk {index}; {location}]\n{document.page_content}")
    return "\n\n".join(context_parts)