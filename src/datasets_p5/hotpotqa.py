"""HotpotQA (distractor) adapter.

Each question ships ~10 context paragraphs and a short answer. We use those
paragraphs as the real evidence and inject one fabricated poison passage.
Downloads from HuggingFace on first use.
"""

from __future__ import annotations

from typing import List

from src.datasets_p5.base import DatasetAdapter, Task
from src.datasets_p5.poison import fake_answer, make_poison_passage

_HF_PATH = "hotpotqa/hotpot_qa"
_HF_CONFIG = "distractor"
_SPLIT = "validation"


def _paragraphs(context: dict) -> List[str]:
    """Flatten HotpotQA's {title, sentences} context into plain passages."""
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    passages: List[str] = []
    for title, sents in zip(titles, sentences):
        body = " ".join(s.strip() for s in sents if s and s.strip())
        if body:
            passages.append(f"{title}. {body}")
    return passages


class HotpotQAAdapter(DatasetAdapter):
    name = "hotpotqa"

    def load(self, n: int) -> List[Task]:
        from datasets import load_dataset

        ds = load_dataset(_HF_PATH, _HF_CONFIG, split=_SPLIT)
        if n > 0:
            ds = ds.select(range(min(n, len(ds))))

        answer_pool = [str(row["answer"]) for row in ds]

        tasks: List[Task] = []
        for i, row in enumerate(ds):
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            real_docs = _paragraphs(row.get("context", {}))
            if not real_docs or not answer:
                continue

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
