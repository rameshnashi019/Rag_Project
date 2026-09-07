"""Precision and recall metrics for retrieval evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalCase:
    """A question and the source files expected to answer it."""

    query: str
    relevant_sources: frozenset[str]

    @classmethod
    def from_sources(cls, query: str, sources: Iterable[str]) -> RetrievalCase:
        """Create a case from source filenames or paths."""
        return cls(query=query, relevant_sources=frozenset(sources))


@dataclass(frozen=True)
class RetrievalScore:
    """Precision and recall for one evaluation question."""

    query: str
    k: int
    relevant_sources: frozenset[str]
    retrieved_sources: tuple[str, ...]
    true_positives: int
    precision_at_k: float
    recall_at_k: float


@dataclass(frozen=True)
class RetrievalSummary:
    """Macro-average retrieval metrics across evaluation questions."""

    questions: int
    mean_precision_at_k: float
    mean_recall_at_k: float


def evaluate_retrieval(
    cases: Sequence[RetrievalCase],
    retrieve_documents: Callable[[str, int], Sequence[Document]],
    *,
    k: int = 4,
) -> list[RetrievalScore]:
    """Evaluate retrieved source files against labeled questions.

    Each source file counts once even when several retrieved chunks come from
    that file. ``precision_at_k`` uses ``k`` as its denominator, while recall
    uses the number of expected source files for the question.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    scores: list[RetrievalScore] = []
    for case in cases:
        if not case.query.strip():
            raise ValueError("evaluation queries must not be empty")
        relevant_sources = _normalize_sources(case.relevant_sources)
        retrieved_sources = _unique_sources(
            document.metadata.get("source") for document in retrieve_documents(case.query, k)
        )
        true_positives = len(set(retrieved_sources) & relevant_sources)
        scores.append(
            RetrievalScore(
                query=case.query,
                k=k,
                relevant_sources=frozenset(relevant_sources),
                retrieved_sources=retrieved_sources,
                true_positives=true_positives,
                precision_at_k=true_positives / k,
                recall_at_k=(
                    true_positives / len(relevant_sources) if relevant_sources else 0.0
                ),
            )
        )
    return scores


def summarize_retrieval(scores: Sequence[RetrievalScore]) -> RetrievalSummary:
    """Calculate macro-average precision and recall from per-question scores."""
    if not scores:
        return RetrievalSummary(
            questions=0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
        )
    return RetrievalSummary(
        questions=len(scores),
        mean_precision_at_k=sum(score.precision_at_k for score in scores) / len(scores),
        mean_recall_at_k=sum(score.recall_at_k for score in scores) / len(scores),
    )


def _normalize_sources(sources: Iterable[str]) -> set[str]:
    return {_source_key(source) for source in sources}


def _unique_sources(sources: Iterable[object]) -> tuple[str, ...]:
    normalized_sources = (_source_key(source) for source in sources if source)
    return tuple(dict.fromkeys(normalized_sources))


def _source_key(source: object) -> str:
    return Path(str(source)).name.casefold()