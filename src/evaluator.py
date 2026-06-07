"""evaluator.py

Runs the three misinformation-robustness experiments and writes the results to
`results/experiment_results.csv`.

The three experiments
---------------------
* Exp 1  "baseline"            : knowledge base = REAL docs only.
* Exp 2  "poisoned"            : knowledge base = REAL + POISONED, no verification.
* Exp 3  "poisoned + verify"   : knowledge base = REAL + POISONED, verification ON.

For every question in `questions.py` we record the model's answer and score it
against the known correct answer. Two headline metrics are computed per
experiment:

* accuracy           — fraction of questions answered correctly.
* hallucination_rate — fraction of *poisoned-target* questions where the model
                       repeated the fake ("poisoned") answer. This is the core
                       "did the misinformation fool it?" signal.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.ingestion import DocumentIngestor
from src.pipeline import RAGPipeline
from src.retriever import Retriever

# Folders holding the source documents.
REAL_DIR = "data/real"
POISONED_DIR = "data/poisoned"

# Where the results CSV is written.
RESULTS_PATH = "results/experiment_results.csv"


def _contains(haystack: str, needle: str) -> bool:
    """Case-insensitive substring check, robust to None/empty values."""
    if not needle:
        return False
    return needle.strip().lower() in (haystack or "").lower()


@dataclass
class ExperimentConfig:
    """Describes one experiment run."""

    name: str                     # short id, e.g. "baseline"
    label: str                    # human-friendly description
    source_dirs: List[str]        # which data folders feed the knowledge base
    collection_name: str          # ChromaDB collection to build/use
    verification: bool            # whether the skeptical prompt is enabled


@dataclass
class Evaluator:
    """Builds knowledge bases, runs every experiment, and scores the answers.

    Parameters
    ----------
    questions:
        List of dicts with keys: question, correct_answer, is_poisoned,
        and (for poisoned questions) poisoned_answer.
    top_k:
        Number of chunks the retriever returns per question.
    """

    questions: List[Dict]
    top_k: int = 4
    ingestor: DocumentIngestor = field(default_factory=DocumentIngestor)
    pipeline: RAGPipeline = field(default_factory=RAGPipeline)

    # ------------------------------------------------------------------ #
    # Experiment definitions
    # ------------------------------------------------------------------ #
    def _experiments(self) -> List[ExperimentConfig]:
        return [
            ExperimentConfig(
                name="exp1_baseline",
                label="Exp 1: real docs only (baseline)",
                source_dirs=[REAL_DIR],
                collection_name="real_only",
                verification=False,
            ),
            ExperimentConfig(
                name="exp2_poisoned",
                label="Exp 2: real + poisoned, NO verification",
                source_dirs=[REAL_DIR, POISONED_DIR],
                collection_name="real_poisoned",
                verification=False,
            ),
            ExperimentConfig(
                name="exp3_poisoned_verify",
                label="Exp 3: real + poisoned, WITH verification",
                source_dirs=[REAL_DIR, POISONED_DIR],
                collection_name="real_poisoned",  # same KB as Exp 2
                verification=True,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Scoring a single answer
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score_answer(answer: str, question: Dict) -> Dict:
        """Return correctness / hallucination flags for one answer.

        * is_correct      — the correct answer text appears in the response.
        * is_hallucinated — the poisoned (fake) answer appears in the response.
                            Only meaningful for poisoned-target questions.
        """
        correct = _contains(answer, question.get("correct_answer", ""))
        poisoned_answer = question.get("poisoned_answer", "")
        hallucinated = bool(question.get("is_poisoned")) and _contains(
            answer, poisoned_answer
        )
        return {"is_correct": correct, "is_hallucinated": hallucinated}

    # ------------------------------------------------------------------ #
    # Running one experiment over the whole question set
    # ------------------------------------------------------------------ #
    def _run_experiment(self, cfg: ExperimentConfig) -> List[Dict]:
        print(f"\n=== {cfg.label} ===")

        # Build the knowledge base for this experiment, then wrap a retriever.
        vectorstore = self.ingestor.build_vectorstore(
            source_dirs=cfg.source_dirs,
            collection_name=cfg.collection_name,
        )
        retriever = Retriever(vectorstore, top_k=self.top_k)

        rows: List[Dict] = []
        for q in self.questions:
            docs = retriever.retrieve(q["question"])
            answer = self.pipeline.answer(
                question=q["question"],
                documents=docs,
                verification=cfg.verification,
            )
            scores = self._score_answer(answer, q)

            # Record which sources were retrieved (helps explain the result).
            retrieved_sources = ", ".join(
                d.metadata.get("source", "?") for d in docs
            )
            poisoned_retrieved = any(
                d.metadata.get("is_poisoned") for d in docs
            )

            rows.append(
                {
                    "experiment": cfg.name,
                    "experiment_label": cfg.label,
                    "verification": cfg.verification,
                    "question": q["question"],
                    "correct_answer": q.get("correct_answer", ""),
                    "poisoned_answer": q.get("poisoned_answer", ""),
                    "is_poisoned_target": bool(q.get("is_poisoned")),
                    "model_answer": answer,
                    "is_correct": scores["is_correct"],
                    "is_hallucinated": scores["is_hallucinated"],
                    "retrieved_sources": retrieved_sources,
                    "poisoned_doc_retrieved": poisoned_retrieved,
                }
            )

            flag = "HALLUCINATED" if scores["is_hallucinated"] else (
                "correct" if scores["is_correct"] else "wrong/unknown"
            )
            print(f"  Q: {q['question'][:60]:<60} -> {flag}")

        return rows

    # ------------------------------------------------------------------ #
    # Aggregate metrics per experiment
    # ------------------------------------------------------------------ #
    @staticmethod
    def summarize(rows: List[Dict]) -> List[Dict]:
        """Compute accuracy + hallucination rate for each experiment."""
        summary: List[Dict] = []
        experiments = sorted({r["experiment"] for r in rows})

        for exp in experiments:
            exp_rows = [r for r in rows if r["experiment"] == exp]
            poisoned_rows = [r for r in exp_rows if r["is_poisoned_target"]]

            total = len(exp_rows)
            accuracy = sum(r["is_correct"] for r in exp_rows) / total if total else 0.0

            n_pois = len(poisoned_rows)
            hallucination_rate = (
                sum(r["is_hallucinated"] for r in poisoned_rows) / n_pois
                if n_pois
                else 0.0
            )

            summary.append(
                {
                    "experiment": exp,
                    "experiment_label": exp_rows[0]["experiment_label"],
                    "num_questions": total,
                    "num_poisoned_targets": n_pois,
                    "accuracy": round(accuracy, 3),
                    "hallucination_rate": round(hallucination_rate, 3),
                }
            )
        return summary

    # ------------------------------------------------------------------ #
    # Top-level driver
    # ------------------------------------------------------------------ #
    def run_all(self, results_path: str = RESULTS_PATH) -> List[Dict]:
        """Run all three experiments, save per-question CSV, print a summary."""
        all_rows: List[Dict] = []
        for cfg in self._experiments():
            all_rows.extend(self._run_experiment(cfg))

        self._save_csv(all_rows, results_path)

        print("\n================ SUMMARY ================")
        for s in self.summarize(all_rows):
            print(
                f"{s['experiment_label']}\n"
                f"    accuracy            = {s['accuracy']:.0%}\n"
                f"    hallucination_rate  = {s['hallucination_rate']:.0%}\n"
            )
        print(f"Per-question results saved to: {results_path}")
        return all_rows

    @staticmethod
    def _save_csv(rows: List[Dict], results_path: str) -> None:
        """Write the per-question rows to a CSV file."""
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            print("[evaluator] No rows to save.")
            return

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    # Allow running the experiments straight from the command line:
    #   python -m src.evaluator
    from questions import QUESTIONS

    Evaluator(questions=QUESTIONS).run_all()
