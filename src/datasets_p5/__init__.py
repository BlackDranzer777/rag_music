"""Dataset adapters for the misinformation-robustness study."""

from __future__ import annotations

from src.datasets_p5.base import DatasetAdapter, Task
from src.datasets_p5.fever import FeverAdapter
from src.datasets_p5.hotpotqa import HotpotQAAdapter
from src.datasets_p5.synthetic import SyntheticAdapter

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
