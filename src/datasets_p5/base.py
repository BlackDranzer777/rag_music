"""Shared Task type and scoring used by every dataset adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LABEL_SUPPORTED = "SUPPORTED"
LABEL_REFUTED = "REFUTED"
LABEL_NEI = "NOT ENOUGH INFO"
FEVER_LABELS = (LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI)


@dataclass
class Task:
    """One evaluation item, normalised so every dataset is handled the same way."""

    id: str
    dataset: str
    question: str
    gold: str
    real_docs: List[str]
    poison_doc: Optional[str] = None
    poison_answer: str = ""
    scoring: str = "substring"
    meta: Dict = field(default_factory=dict)

    @property
    def is_poisoned(self) -> bool:
        return bool(self.poison_doc)


def _contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return needle.strip().lower() in (haystack or "").lower()


def parse_label(answer: str) -> str:
    """Extract a FEVER verdict from a free-text answer.

    NEI is checked first so phrases like "do not support" fall through to NEI
    instead of being misread as SUPPORTED.
    """
    text = (answer or "").lower()

    nei_markers = (
        "not enough info",
        "notenoughinfo",
        "insufficient",
        "cannot be determined",
        "can't be determined",
        "no evidence",
        "unverifiable",
    )
    if any(m in text for m in nei_markers):
        return LABEL_NEI

    refute_markers = ("refut", "false", "contradict", "incorrect", "disproven")
    support_markers = ("support", "true", "correct", "confirm", "verified")

    refuted = any(m in text for m in refute_markers)
    supported = any(m in text for m in support_markers)

    if refuted and not supported:
        return LABEL_REFUTED
    if supported and not refuted:
        return LABEL_SUPPORTED
    return LABEL_NEI


def score_answer(answer: str, task: Task) -> Dict[str, bool]:
    """Flag whether an answer matches the gold answer and/or the poison answer."""
    if task.scoring == "label":
        predicted = parse_label(answer)
        is_correct = predicted == task.gold.upper()
        is_hallucinated = bool(task.poison_answer) and predicted == task.poison_answer.upper()
        return {"is_correct": is_correct, "is_hallucinated": is_hallucinated}

    is_correct = _contains(answer, task.gold)
    is_hallucinated = bool(task.poison_answer) and _contains(answer, task.poison_answer)
    return {"is_correct": is_correct, "is_hallucinated": is_hallucinated}


class DatasetAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def load(self, n: int) -> List[Task]:
        """Return up to n normalised Task items for this dataset."""
        raise NotImplementedError

    def score(self, answer: str, task: Task) -> Dict[str, bool]:
        return score_answer(answer, task)
