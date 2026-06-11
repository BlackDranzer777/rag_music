"""hotpotqa.py

HotpotQA adapter (P5 Dataset bullet 2: *multi-hop QA benchmark for evidence
reasoning*).

We use the ``distractor`` configuration: each question ships 10 paragraphs — a
couple of gold "supporting" paragraphs plus distractors — and a short answer.
That self-contained context maps directly onto our per-item knowledge base:

* ``real_docs``    — the 10 context paragraphs (distractors are legitimate
                     noise, not poison).
* ``poison_doc``   — a fabricated passage asserting a false answer.
* ``gold``         — the short answer (substring-scored).

The dataset downloads from HuggingFace on first use.
"""

from __future__ import annotations

from typing import List

from src.datasets_p5.base import DatasetAdapter, Task
from src.datasets_p5.poison import fake_answer, make_poison_passage

# Newer `datasets`/`huggingface_hub` require a namespaced repo id.
_HF_PATH = "hotpotqa/hotpot_qa"
_HF_CONFIG = "distractor"
_SPLIT = "validation"


def _paragraphs(context: dict) -> List[str]:
    """Flatten HotpotQA's {title: [...], sentences: [[...]]} into passages."""
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    passages: List[str] = []
    for title, sents in zip(titles, sentences):
        body = " ".join(s.strip() for s in sents if s and s.strip())
        if body:
            passages.append(f"{title}. {body}")
    return passages


class HotpotQAAdapter(DatasetAdapter):
    """Loads HotpotQA (distractor) and injects a poison answer per question."""

    name = "hotpotqa"

    def load(self, n: int) -> List[Task]:
        # Imported lazily so the offline (synthetic) path never needs `datasets`.
        from datasets import load_dataset

        ds = load_dataset(_HF_PATH, _HF_CONFIG, split=_SPLIT)
        if n > 0:
            ds = ds.select(range(min(n, len(ds))))

        # Pool of real answers, used to make non-numeric poison answers plausible.
        answer_pool = [str(row["answer"]) for row in ds]

        tasks: List[Task] = []
        for i, row in enumerate(ds):
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            real_docs = _paragraphs(row.get("context", {}))
            if not real_docs or not answer:
                continue  # skip malformed rows

            false_answer = fake_answer(answer, pool=answer_pool)
            poison_doc = make_poison_passage(
                subject=question,
                false_answer=false_answer,
                style="fact",
            )

            tasks.append(
                Task(
                    id=str(row.get("id", f"hotpotqa-{i}")),
                    dataset=self.name,
                    question=question,
                    gold=answer,
                    real_docs=real_docs,
                    poison_doc=poison_doc,
                    poison_answer=false_answer,
                    scoring="substring",
                    meta={"type": row.get("type"), "level": row.get("level")},
                )
            )
        return tasks
