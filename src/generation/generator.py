"""Generate grounded answers from retrieved document context."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from reterieval import RetrievalResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You answer questions using only the provided PDF context. "
    "If the context does not contain the answer, say you do not know. "
    "Cite the source and page shown in the context when answering."
)


def generate_answer(
    question: str,
    retrieval_result: RetrievalResult,
    *,
    model: str | None = None,
    chat_model: BaseChatModel | None = None,
) -> str:
    """Generate an answer from retrieved PDF context."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not retrieval_result.context.strip():
        logger.warning("Cannot generate an answer without retrieved context")
        return "I could not find relevant information in the documents."

    try:
        llm = chat_model or _create_chat_model(model)
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"PDF context:\n{retrieval_result.context}\n\n"
                        f"Question: {question}"
                    )
                ),
            ]
        )
        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer)
        logger.info("Generated answer using %d retrieved chunk(s)", len(retrieval_result.documents))
        return answer.strip()
    except Exception:
        logger.exception("Failed to generate answer")
        return "I could not generate an answer from the documents."


def _create_chat_model(model: str | None) -> ChatOpenAI:
    load_dotenv()
    chat_model = model or os.getenv("OPENAI_CHAT_MODEL")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in .env when using OpenAI")
    if not chat_model:
        raise ValueError("OPENAI_CHAT_MODEL must be set in .env when using OpenAI")
    return ChatOpenAI(model=chat_model, temperature=0)