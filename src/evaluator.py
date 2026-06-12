"""Run the misinformation-robustness experiments and write results to results/.

For each dataset (synthetic, HotpotQA, FEVER) we run three experiments:
    Exp 1  baseline          - real evidence only.
    Exp 2  poisoned          - real + a fabricated poison passage, no verification.
    Exp 3  poisoned + verify  - real + poison, with the verification prompt.

Each item gets its own in-memory knowledge base, so retrieval stays relevant and
the runs are isolated. The evaluator can sweep models and top_k values, and
optionally measure self-consistency.
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

RESULTS_PATH = "results/experiment_results.csv"
SUMMARY_PATH = "results/experiment_summary.csv"

DEFAULT_MODELS = ["llama-3.3-70b-versatile"]
DEFAULT_TOP_KS = [4]


def _fresh_collection_name() -> str:
    # UUID keeps each ephemeral collection name unique and Chroma-valid.
    return f"c{uuid.uuid4().hex}"


@dataclass
class ExperimentConfig:
    name: str
    label: str
    include_poison: bool
    verification: bool


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
    _pipelines: Dict = field(default_factory=dict, init=False, repr=False)

    def _pipeline(self, model: str, temperature: float = 0.0) -> RAGPipeline:
        key = (model, temperature)
        if key not in self._pipelines:
            self._pipelines[key] = RAGPipeline(model_name=model, temperature=temperature)
        return self._pipelines[key]

    @staticmethod
    def _task_documents(task: Task, include_poison: bool) -> List[Document]:
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
        docs = self._task_documents(task, cfg.include_poison)
        vectorstore = self.ingestor.build_vectorstore_from_documents(
            docs, collection_name=_fresh_collection_name()
        )
        return vectorstore.similarity_search(task.question, k=top_k)

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
                    "self_consistency": "",
                }
            )

            flag = "HALLUCINATED" if scores["is_hallucinated"] else (
                "correct" if scores["is_correct"] else "wrong/unknown"
            )
            print(f"  {task.id[:24]:<24} -> {flag}")

        return rows

    def _self_consistency(
        self,
        adapter: DatasetAdapter,
        tasks: List[Task],
        cfg: ExperimentConfig,
        model: str,
        top_k: int,
    ) -> float:
        """Resample each item a few times and return mean agreement in [0, 1]."""
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
    return re.sub(r"\s+", " ", (answer or "").strip().lower())[:200]


if __name__ == "__main__":
    Evaluator(
        n_per_dataset=10,
        models=DEFAULT_MODELS,
        top_ks=DEFAULT_TOP_KS,
        self_consistency_samples=0,
    ).run_all()
