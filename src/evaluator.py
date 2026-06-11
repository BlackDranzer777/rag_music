"""evaluator.py

Runs the P5 misinformation-robustness experiments across the three datasets
(synthetic, HotpotQA, FEVER) and writes per-question results plus an aggregate
summary to ``results/``.

For every dataset we run three experiments — the same story as the original
demo, now applied to real benchmarks:

* Exp 1  "baseline"          : knowledge base = the item's REAL evidence only.
* Exp 2  "poisoned"          : REAL + a fabricated POISON passage, no verification.
* Exp 3  "poisoned + verify" : REAL + POISON, with the skeptical verification prompt.

Each item gets its **own** tiny, in-memory knowledge base (its evidence
passages, plus the poison doc in Exp 2/3), so retrieval stays relevant and the
runs are isolated and fast.

P5 also asks for the metrics to be compared **across models and retrieval
settings**, and for a **self-consistency** measure, so the evaluator sweeps a
list of models and ``top_k`` values and can resample answers to estimate
agreement.

Headline metrics per (dataset, experiment, model, top_k):

* accuracy            — fraction of items answered correctly.
* hallucination_rate  — fraction of *poisoned-target* items where the model
                        repeated the fake answer/label (did the poison win?).
* self_consistency    — mean agreement of resampled answers (optional pass).
"""

from __future__ import annotations

import csv
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.datasets_p5 import ADAPTERS
from src.datasets_p5.base import DatasetAdapter, Task
from src.ingestion import DocumentIngestor
from src.pipeline import RAGPipeline

# Where results are written.
RESULTS_PATH = "results/experiment_results.csv"
SUMMARY_PATH = "results/experiment_summary.csv"

# Default sweep — kept small so a first run finishes quickly. Override in
# __main__ or when constructing the Evaluator.
DEFAULT_MODELS = ["llama-3.3-70b-versatile"]
DEFAULT_TOP_KS = [4]


def _fresh_collection_name() -> str:
    """A unique, always-valid Chroma collection name for an ephemeral KB.

    Each per-item knowledge base is in-memory and short-lived, so the name only
    needs to be unique and satisfy Chroma's rules (alphanumeric ends, 3-512
    chars from [a-zA-Z0-9._-]). A UUID guarantees both.
    """
    return f"c{uuid.uuid4().hex}"


@dataclass
class ExperimentConfig:
    """Describes one experiment variant (independent of dataset)."""

    name: str            # short id, e.g. "exp2_poisoned"
    label: str           # human-friendly description
    include_poison: bool # add the poison doc to the knowledge base?
    verification: bool   # use the skeptical verification prompt?


EXPERIMENTS = [
    ExperimentConfig(
        name="exp1_baseline",
        label="Exp 1: real evidence only (baseline)",
        include_poison=False,
        verification=False,
    ),
    ExperimentConfig(
        name="exp2_poisoned",
        label="Exp 2: real + poisoned, NO verification",
        include_poison=True,
        verification=False,
    ),
    ExperimentConfig(
        name="exp3_poisoned_verify",
        label="Exp 3: real + poisoned, WITH verification",
        include_poison=True,
        verification=True,
    ),
]


@dataclass
class Evaluator:
    """Drives the dataset adapters through every experiment and scores answers.

    Parameters
    ----------
    adapters:
        Dataset adapters to evaluate (default: all three).
    n_per_dataset:
        How many items to load from each dataset.
    models:
        Groq model ids to compare.
    top_ks:
        Retrieval depths to compare.
    self_consistency_samples:
        If > 0, run an extra pass that resamples each answer this many times at
        ``sc_temperature`` and records the mean agreement. 0 disables it.
    sc_temperature / self_consistency_cap:
        Sampling temperature and the max items per group used for the
        self-consistency pass (capped to bound cost).
    """

    adapters: List[DatasetAdapter] = field(
        default_factory=lambda: [cls() for cls in ADAPTERS.values()]
    )
    n_per_dataset: int = 10
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    top_ks: List[int] = field(default_factory=lambda: list(DEFAULT_TOP_KS))
    self_consistency_samples: int = 0
    sc_temperature: float = 0.7
    self_consistency_cap: int = 10

    ingestor: DocumentIngestor = field(default_factory=DocumentIngestor)
    # Cache of pipelines keyed by (model_name, temperature).
    _pipelines: Dict = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Pipeline / knowledge-base helpers
    # ------------------------------------------------------------------ #
    def _pipeline(self, model: str, temperature: float = 0.0) -> RAGPipeline:
        key = (model, temperature)
        if key not in self._pipelines:
            self._pipelines[key] = RAGPipeline(model_name=model, temperature=temperature)
        return self._pipelines[key]

    @staticmethod
    def _task_documents(task: Task, include_poison: bool) -> List[Document]:
        """Build the LangChain documents that form this item's knowledge base."""
        docs = [
            Document(
                page_content=text,
                metadata={"source": f"{task.dataset}_real_{j}", "is_poisoned": False},
            )
            for j, text in enumerate(task.real_docs)
        ]
        if include_poison and task.poison_doc:
            docs.append(
                Document(
                    page_content=task.poison_doc,
                    metadata={"source": f"{task.dataset}_POISON", "is_poisoned": True},
                )
            )
        return docs

    def _retrieve(self, task: Task, cfg: ExperimentConfig, top_k: int):
        """Build the per-item KB and return the top_k retrieved documents."""
        docs = self._task_documents(task, cfg.include_poison)
        vectorstore = self.ingestor.build_vectorstore_from_documents(
            docs, collection_name=_fresh_collection_name()
        )
        retrieved = vectorstore.similarity_search(task.question, k=top_k)
        # In-memory collections are GC'd with the vectorstore; nothing persists.
        return retrieved

    # ------------------------------------------------------------------ #
    # Running one (dataset, experiment, model, top_k) combination
    # ------------------------------------------------------------------ #
    def _run_combo(
        self,
        adapter: DatasetAdapter,
        tasks: List[Task],
        cfg: ExperimentConfig,
        model: str,
        top_k: int,
    ) -> List[Dict]:
        pipeline = self._pipeline(model, temperature=0.0)
        print(f"\n=== [{adapter.name}] {cfg.label} | model={model} top_k={top_k} ===")

        rows: List[Dict] = []
        for task in tasks:
            task_type = "label" if task.scoring == "label" else "qa"
            docs = self._retrieve(task, cfg, top_k)
            answer = pipeline.answer(
                question=task.question,
                documents=docs,
                verification=cfg.verification,
                task_type=task_type,
            )
            scores = adapter.score(answer, task)

            poison_retrieved = any(d.metadata.get("is_poisoned") for d in docs)
            rows.append(
                {
                    "dataset": adapter.name,
                    "experiment": cfg.name,
                    "experiment_label": cfg.label,
                    "model": model,
                    "top_k": top_k,
                    "verification": cfg.verification,
                    "question": task.question,
                    "gold": task.gold,
                    "poison_answer": task.poison_answer,
                    "is_poisoned_target": task.is_poisoned,
                    "model_answer": answer,
                    "is_correct": scores["is_correct"],
                    "is_hallucinated": scores["is_hallucinated"],
                    "poison_doc_retrieved": poison_retrieved,
                    "self_consistency": "",  # filled by the optional SC pass
                }
            )

            flag = "HALLUCINATED" if scores["is_hallucinated"] else (
                "correct" if scores["is_correct"] else "wrong/unknown"
            )
            print(f"  {task.id[:24]:<24} -> {flag}")

        return rows

    # ------------------------------------------------------------------ #
    # Optional self-consistency pass
    # ------------------------------------------------------------------ #
    def _self_consistency(
        self,
        adapter: DatasetAdapter,
        tasks: List[Task],
        cfg: ExperimentConfig,
        model: str,
        top_k: int,
    ) -> float:
        """Mean agreement of resampled answers for one group of items.

        For each (capped) item we ask the model ``self_consistency_samples``
        times at a non-zero temperature and compute the fraction of answers
        equal to the most common one. Averaged over items, this is a simple
        self-consistency score in [0, 1] (1.0 = perfectly stable).
        """
        pipeline = self._pipeline(model, temperature=self.sc_temperature)
        sample_tasks = tasks[: self.self_consistency_cap]
        if not sample_tasks:
            return 0.0

        per_item: List[float] = []
        for task in sample_tasks:
            task_type = "label" if task.scoring == "label" else "qa"
            docs = self._retrieve(task, cfg, top_k)
            answers = [
                _normalise_answer(
                    pipeline.answer(
                        question=task.question,
                        documents=docs,
                        verification=cfg.verification,
                        task_type=task_type,
                    )
                )
                for _ in range(self.self_consistency_samples)
            ]
            counts = Counter(answers)
            modal = counts.most_common(1)[0][1]
            per_item.append(modal / len(answers))

        return round(sum(per_item) / len(per_item), 3)

    # ------------------------------------------------------------------ #
    # Top-level driver
    # ------------------------------------------------------------------ #
    def run_all(
        self,
        results_path: str = RESULTS_PATH,
        summary_path: str = SUMMARY_PATH,
    ) -> List[Dict]:
        # Load each dataset once and reuse the items across every combination.
        loaded: List[tuple] = []
        for adapter in self.adapters:
            print(f"[evaluator] loading {adapter.name} (n={self.n_per_dataset}) ...")
            tasks = adapter.load(self.n_per_dataset)
            print(f"[evaluator]   -> {len(tasks)} items")
            if tasks:
                loaded.append((adapter, tasks))

        all_rows: List[Dict] = []
        for adapter, tasks in loaded:
            for model in self.models:
                for top_k in self.top_ks:
                    for cfg in EXPERIMENTS:
                        rows = self._run_combo(adapter, tasks, cfg, model, top_k)

                        sc_value = ""
                        if self.self_consistency_samples > 0:
                            sc_value = self._self_consistency(
                                adapter, tasks, cfg, model, top_k
                            )
                            for r in rows:
                                r["self_consistency"] = sc_value

                        all_rows.extend(rows)

        self._save_csv(all_rows, results_path)
        summary = self.summarize(all_rows)
        self._save_csv(summary, summary_path)
        self._print_summary(summary)
        print(f"\nPer-question results -> {results_path}")
        print(f"Aggregate summary    -> {summary_path}")
        return all_rows

    # ------------------------------------------------------------------ #
    # Aggregation / output
    # ------------------------------------------------------------------ #
    @staticmethod
    def summarize(rows: List[Dict]) -> List[Dict]:
        """Accuracy + hallucination rate per (dataset, experiment, model, top_k)."""
        summary: List[Dict] = []
        keys = sorted({
            (r["dataset"], r["experiment"], r["model"], r["top_k"]) for r in rows
        })
        for dataset, exp, model, top_k in keys:
            group = [
                r for r in rows
                if (r["dataset"], r["experiment"], r["model"], r["top_k"])
                == (dataset, exp, model, top_k)
            ]
            poisoned = [r for r in group if r["is_poisoned_target"]]
            total = len(group)
            accuracy = sum(r["is_correct"] for r in group) / total if total else 0.0
            n_pois = len(poisoned)
            hallucination_rate = (
                sum(r["is_hallucinated"] for r in poisoned) / n_pois if n_pois else 0.0
            )
            summary.append(
                {
                    "dataset": dataset,
                    "experiment": exp,
                    "experiment_label": group[0]["experiment_label"],
                    "model": model,
                    "top_k": top_k,
                    "num_questions": total,
                    "num_poisoned_targets": n_pois,
                    "accuracy": round(accuracy, 3),
                    "hallucination_rate": round(hallucination_rate, 3),
                    "self_consistency": group[0].get("self_consistency", ""),
                }
            )
        return summary

    @staticmethod
    def _print_summary(summary: List[Dict]) -> None:
        print("\n================ SUMMARY ================")
        for s in summary:
            sc = s["self_consistency"]
            sc_str = f"  self_consistency={sc}" if sc != "" else ""
            print(
                f"[{s['dataset']}] {s['experiment']} "
                f"(model={s['model']}, top_k={s['top_k']}): "
                f"accuracy={s['accuracy']:.0%}  "
                f"hallucination={s['hallucination_rate']:.0%}{sc_str}"
            )

    @staticmethod
    def _save_csv(rows: List[Dict], path_str: str) -> None:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            print(f"[evaluator] No rows to save for {path_str}.")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _normalise_answer(answer: str) -> str:
    """Collapse an answer to a comparable key for self-consistency counting."""
    return re.sub(r"\s+", " ", (answer or "").strip().lower())[:200]


if __name__ == "__main__":
    # Quick start:  python -m src.evaluator
    #
    # Defaults are intentionally small (one model, top_k=4, 10 items/dataset,
    # self-consistency off) so the first run finishes fast. Scale these up once
    # a small run looks correct.
    Evaluator(
        n_per_dataset=10,
        models=DEFAULT_MODELS,
        top_ks=DEFAULT_TOP_KS,
        self_consistency_samples=0,
    ).run_all()
