"""Export the evaluator datasets to .txt files for the Streamlit app.

For each dataset (synthetic / HotpotQA / FEVER) every item is written as one real
.txt (its evidence passages) and, when poisoned, one poison .txt — the same
real/poisoned layout the app already uses for the artist files:

    data/<dataset>/real/<id>.txt
    data/<dataset>/poisoned/<id>.txt
    data/<dataset>/index.csv        # id, question, gold, poison_answer

Run with:  python -m src.export_datasets
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List

from src.datasets_p5 import ADAPTERS
from src.datasets_p5.base import Task

DATA_ROOT = Path("data")


def _safe(name: str) -> str:
    """Make a Task id safe to use as a filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80]


def export_dataset(name: str, n: int) -> int:
    tasks: List[Task] = ADAPTERS[name]().load(n)
    if not tasks:
        print(f"[export] {name}: no items loaded, skipping.")
        return 0

    real_dir = DATA_ROOT / name / "real"
    poison_dir = DATA_ROOT / name / "poisoned"
    real_dir.mkdir(parents=True, exist_ok=True)
    poison_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for task in tasks:
        stem = _safe(task.id)

        real_text = "\n\n".join(task.real_docs).strip()
        (real_dir / f"{stem}.txt").write_text(real_text, encoding="utf-8")

        if task.poison_doc:
            (poison_dir / f"{stem}.txt").write_text(
                task.poison_doc.strip(), encoding="utf-8"
            )

        index_rows.append(
            {
                "id": task.id,
                "question": task.question,
                "gold": task.gold,
                "poison_answer": task.poison_answer,
                "scoring": task.scoring,
            }
        )

    with (DATA_ROOT / name / "index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    n_poison = sum(1 for t in tasks if t.poison_doc)
    print(
        f"[export] {name}: {len(tasks)} items -> {real_dir} "
        f"({n_poison} poison files in {poison_dir})"
    )
    return len(tasks)


def export_all(n: int = 10) -> None:
    total = 0
    for name in ADAPTERS:
        total += export_dataset(name, n)
    print(f"\nDone. Exported {total} items across {len(ADAPTERS)} datasets to {DATA_ROOT}/.")


if __name__ == "__main__":
    export_all(n=10)
