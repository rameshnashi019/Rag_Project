"""OpenAI embeddings with a local Hugging Face fallback."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class FallbackEmbeddings(Embeddings):
    """Use the fallback provider when the primary provider fails."""

    def __init__(self, primary: Embeddings, fallback_factory: Callable[[], Embeddings]) -> None:
        self.primary = primary
        self.fallback_factory = fallback_factory
        self.fallback: Embeddings | None = None
        self._using_fallback = False

    def _get_fallback(self) -> Embeddings:
        if self.fallback is None:
            logger.info("Initializing local fallback embedding model")
            self.fallback = self.fallback_factory()
        return self.fallback

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            if not self._using_fallback:
                return self.primary.embed_documents(texts)
        except Exception:
            logger.exception("Primary embedding provider failed; using fallback")
            self._using_fallback = True
        return self._get_fallback().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            if not self._using_fallback:
                return self.primary.embed_query(text)
        except Exception:
            logger.exception("Primary query embedding failed; using fallback")
            self._using_fallback = True
        return self._get_fallback().embed_query(text)


def create_embeddings(
    *,
    model: str | None = None,
    fallback_model: str | None = None,
    dimensions: int | None = None,
) -> Embeddings:
    """Create OpenAI embeddings with a local fallback."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = model or os.getenv("OPENAI_EMBEDDING_MODEL")
    hf_model = fallback_model or os.getenv(
        "HF_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    fallback_factory = lambda: HuggingFaceEmbeddings(model_name=hf_model)
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY is missing; using local Hugging Face embeddings")
        return fallback_factory()
    if not openai_model:
        raise ValueError("OPENAI_EMBEDDING_MODEL must be set in .env when using OpenAI")

    openai_options: dict[str, object] = {
        "model": openai_model,
        "api_key": openai_api_key,
    }
    if dimensions is not None:
        openai_options["dimensions"] = dimensions

    try:
        primary = OpenAIEmbeddings(**openai_options)
        logger.info("Using OpenAI embedding model: %s", openai_model)
        return FallbackEmbeddings(primary, fallback_factory)
    except Exception:
        logger.exception("Could not initialize OpenAI embeddings; using fallback")
        return fallback_factory()