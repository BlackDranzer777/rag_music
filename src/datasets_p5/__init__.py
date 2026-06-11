"""datasets_p5

Dataset adapters for the P5 "In RAG We Trust?" misinformation-robustness study.

Each adapter turns a benchmark (HotpotQA, FEVER) or a generator (synthetic) into
a common list of :class:`~src.datasets_p5.base.Task` objects that the evaluator
can run uniformly: build a per-item knowledge base, retrieve, ask the LLM, and
score the answer.

The three adapters intentionally share one interface so the evaluator never has
to know which benchmark it is running.
"""

from __future__ import annotations

from src.datasets_p5.base import DatasetAdapter, Task
from src.datasets_p5.fever import FeverAdapter
from src.datasets_p5.hotpotqa import HotpotQAAdapter
from src.datasets_p5.synthetic import SyntheticAdapter

# Registry so the evaluator / CLI can select adapters by short name.
ADAPTERS = {
    "synthetic": SyntheticAdapter,
    "hotpotqa": HotpotQAAdapter,
    "fever": FeverAdapter,
}

__all__ = [
    "Task",
    "DatasetAdapter",
    "SyntheticAdapter",
    "HotpotQAAdapter",
    "FeverAdapter",
    "ADAPTERS",
]
