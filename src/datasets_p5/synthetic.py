"""synthetic.py

Synthetic dataset (P5 Dataset bullet 3: *deliberately injected false /
contradictory passages*).

This adapter needs no download and is fully deterministic, so the offline path
always works and the experiment story is clean: every item has exactly one real
passage and one contradicting poison passage, with short, distinctive answers
that substring-match reliably.

Each fact is an (entity, attribute, value) triple turned into:

* a question        — "What is the {attribute} of {entity}?"
* a real passage    — states the true value.
* a poison passage  — states a different, false value.
"""

from __future__ import annotations

from typing import List

from src.datasets_p5.base import DatasetAdapter, Task
from src.datasets_p5.poison import fake_answer, make_poison_passage

# A pool of self-contained facts. Values are short and distinctive (a year, a
# city, a number) so they substring-match cleanly in a free-form answer.
# (entity, attribute, value)
_FACTS = [
    ("the band Quantum Echo", "formation year", "2011"),
    ("the artist Luna Vega", "home city", "Lisbon"),
    ("the Aurora Bridge", "opening year", "1932"),
    ("the novel Glass Harbor", "publication year", "1987"),
    ("the painter Mira Kovac", "birth city", "Zagreb"),
    ("the satellite Helios-4", "launch year", "2009"),
    ("the Verde Tower", "height in metres", "412"),
    ("the chess champion Ivan Petrov", "title year", "1996"),
    ("the film Silent Cartography", "release year", "2003"),
    ("the river Calden", "length in kilometres", "318"),
    ("the company Northwind Labs", "founding year", "2007"),
    ("the marathon record holder Sara Lindqvist", "record time city", "Berlin"),
    ("the observatory Mauna Cielo", "altitude in metres", "3050"),
    ("the orchestra Camerata Nord", "founding city", "Oslo"),
    ("the spacecraft Vela-2", "mission duration in days", "287"),
    ("the cathedral of Saint Aldric", "completion year", "1456"),
    ("the videogame Echo Protocol", "release year", "2015"),
    ("the vineyard Castel Rosso", "founding year", "1894"),
    ("the lighthouse at Cape Miren", "height in metres", "47"),
    ("the festival Lumina", "first edition year", "1998"),
]


class SyntheticAdapter(DatasetAdapter):
    """Generates self-contained, poisoned (entity, attribute, value) facts."""

    name = "synthetic"

    def load(self, n: int) -> List[Task]:
        facts = _FACTS[: max(0, n)]
        # The pool of all true values lets the poison swap in a plausible
        # alternative for non-numeric answers.
        value_pool = [v for _, _, v in _FACTS]

        tasks: List[Task] = []
        for i, (entity, attribute, value) in enumerate(facts):
            question = f"What is the {attribute} of {entity}?"
            false_value = fake_answer(value, pool=value_pool)

            real_doc = (
                f"Fact sheet: {entity}. "
                f"The {attribute} of {entity} is {value}. "
                f"This is well documented in standard references."
            )
            poison_doc = make_poison_passage(
                subject=f"the {attribute} of {entity}",
                false_answer=false_value,
                style="fact",
            )

            tasks.append(
                Task(
                    id=f"synthetic-{i}",
                    dataset=self.name,
                    question=question,
                    gold=value,
                    real_docs=[real_doc],
                    poison_doc=poison_doc,
                    poison_answer=false_value,
                    scoring="substring",
                    meta={"entity": entity, "attribute": attribute},
                )
            )
        return tasks
