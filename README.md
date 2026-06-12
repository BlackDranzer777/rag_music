# 🛡️ RAG Misinformation Robustness Tester (P5: *In RAG We Trust?*)

A Retrieval-Augmented Generation (RAG) project for a university NLP course. It
answers questions / verifies claims from a knowledge base, then tests **how
easily the system is fooled by injected fake ("poisoned") documents** — and
whether a *verification prompt* makes it more robust.

It evaluates three datasets behind one interface:

| Dataset       | What it provides                          | Scoring        |
| ------------- | ----------------------------------------- | -------------- |
| **synthetic** | self-contained injected contradictions (offline, deterministic) | substring |
| **HotpotQA**  | multi-hop QA with distractor paragraphs   | substring      |
| **FEVER**     | fact-checking claims + gold evidence      | label (SUPPORTED/REFUTED/NEI) |

HotpotQA and FEVER download from HuggingFace on first use; synthetic needs no
network. Everything else runs on **free** tools:

| Concern        | Choice                                                |
| -------------- | ----------------------------------------------------- |
| Embeddings     | `all-MiniLM-L6-v2` via sentence-transformers (LOCAL)  |
| Vector store   | ChromaDB (local, on disk)                             |
| LLM            | `llama-3.3-70b-versatile` via Groq (free tier)        |
| UI             | Streamlit                                             |

> The **only** thing that touches the internet is the Groq LLM call.
> Embeddings are 100% local — no OpenAI, no paid APIs.

---

## 📁 Project structure

```
music-rag/
├── data/
│   ├── real/          # real .txt files for the Streamlit UI demo (artists)
│   ├── poisoned/      # fake .txt files for the Streamlit UI demo (artists)
│   └── <dataset>/     # exported benchmark items (synthetic/hotpotqa/fever)
│       ├── real/      #   one .txt per item's real evidence
│       ├── poisoned/  #   one .txt per item's poison passage
│       └── index.csv  #   id, question, gold, poison_answer
├── src/
│   ├── ingestion.py        # embed + store in ChromaDB (folders OR in-memory docs)
│   ├── retriever.py        # question -> top-k relevant chunks
│   ├── pipeline.py         # question/claim + chunks -> Groq LLM -> answer (+verify, +label mode)
│   ├── evaluator.py        # runs experiments across datasets/models/top-k, saves CSV
│   ├── export_datasets.py  # dumps the datasets to data/<dataset>/ .txt files for the UI
│   └── datasets_p5/   # the three dataset adapters behind one interface
│       ├── base.py        # Task dataclass, DatasetAdapter ABC, shared scoring
│       ├── poison.py      # controlled data-poisoning helpers
│       ├── synthetic.py   # offline injected-contradiction generator
│       ├── hotpotqa.py    # HotpotQA (distractor) loader
│       └── fever.py       # FEVER gold-evidence loader (label scoring)
├── notebooks/
│   └── demo.ipynb     # loads results CSV, charts per dataset / model
├── results/           # experiment_results.csv + experiment_summary.csv land here
├── questions.py       # legacy question set (Streamlit UI demo only)
├── app.py             # Streamlit UI
├── .env               # GROQ_API_KEY=...
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # macOS / Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your free Groq API key** to the `.env` file:

   ```
   GROQ_API_KEY=your_key_here
   ```

   Get one at <https://console.groq.com/keys>.

4. **(Optional) Add your own data.** Drop real artist `.txt` files into
   `data/real/` and fake ones into `data/poisoned/`. Example files are already
   provided so you can see the expected format. If you add questions, edit
   `questions.py` to match (see the format described in that file).

---

## ▶️ Running things

### The interactive UI

```bash
streamlit run app.py
```

Then in the browser:
- pick a **Data source** — the hand-written artist files, or one of the
  exported benchmark datasets (synthetic / hotpotqa / fever),
- toggle **"Include poisoned documents"** to add/remove the fake sources,
- toggle **"Verification mode"** to enable the skeptical, contradiction-checking
  prompt,
- ask a question and inspect both the answer and the retrieved sources
  (poisoned sources are flagged ☠️).

The artist files work out of the box. To make the benchmark datasets selectable,
export them to `.txt` first (see below).

### Exporting the datasets for the UI

```bash
python -m src.export_datasets
```

This writes each dataset's items to `data/<dataset>/real/` and
`data/<dataset>/poisoned/` (one `.txt` per item), plus a `data/<dataset>/index.csv`
listing each `question`, its `gold` answer, and the `poison_answer`. The Streamlit
app then reads those folders just like the artist files. **First run downloads
HotpotQA and FEVER from HuggingFace** (synthetic is offline). The app and the
exporter run independently — exporting does not run any experiments.

### The experiments (generates the CSV)

```bash
python -m src.evaluator
```

This runs the three experiments over all three datasets (synthetic, HotpotQA,
FEVER) and writes:

- `results/experiment_results.csv` — one row per item, with `dataset`, `model`,
  `top_k`, `is_correct`, `is_hallucinated`, `self_consistency`, …
- `results/experiment_summary.csv` — accuracy / hallucination_rate /
  self_consistency aggregated per (dataset, experiment, model, top_k).

**First run downloads HotpotQA and FEVER from HuggingFace.** Defaults are small
(10 items/dataset, one model, `top_k=4`, self-consistency off) so it finishes
quickly. To scale up or sweep models / retrieval settings / self-consistency,
edit the `Evaluator(...)` call at the bottom of `src/evaluator.py`, e.g.:

```python
Evaluator(
    n_per_dataset=50,
    models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    top_ks=[2, 4, 8],
    self_consistency_samples=5,   # 0 = off
).run_all()
```

### The charts

Open `notebooks/demo.ipynb` and run all cells to visualise accuracy and
hallucination rate per experiment.

---

## 🧪 The three experiments

| #   | Knowledge base        | Verification prompt | What it shows                          |
| --- | --------------------- | ------------------- | -------------------------------------- |
| 1   | Real docs only        | off                 | Baseline accuracy (no misinformation)  |
| 2   | Real + Poisoned       | off                 | How badly fake docs fool the model     |
| 3   | Real + Poisoned       | **on**              | Whether verification reduces the damage |

**Metrics** (printed and saved per dataset × experiment × model × top_k):

- **accuracy** — fraction of items answered correctly (substring match for
  QA datasets; correct SUPPORTED/REFUTED/NEI label for FEVER).
- **hallucination_rate** — fraction of poisoned-target items where the model
  repeated the *fake* answer / flipped verdict. The core "did the
  misinformation win?" number.
- **self_consistency** *(optional)* — mean agreement of resampled answers at a
  non-zero temperature; how reliably the model gives the same answer.

The expected story: hallucination_rate jumps from Exp 1 → Exp 2, then drops
again in Exp 3 if verification helps. P5 also asks you to compare these across
**models** and **retrieval settings** — hence the `models` / `top_ks` sweep.

---

## 🧠 How it works (quick tour for the report)

1. **Datasets** (`datasets_p5/`) each load into a common `Task` (question/claim,
   real evidence passages, one fabricated poison passage, gold answer/label).
   `poison.py` builds the poison: it flips yes/no, perturbs numbers, or swaps in
   a plausible alternative entity.
2. **Ingestion** (`ingestion.py`) builds a **per-item, in-memory** ChromaDB
   collection from that item's passages (plus the poison doc in Exp 2/3),
   embedding each chunk locally with MiniLM. Real vs poisoned origin is kept in
   the metadata.
3. **Retrieval** (`retriever.py`) embeds the question and returns the top-k most
   similar chunks by cosine similarity.
4. **Generation** (`pipeline.py`) stuffs those chunks into a prompt and calls the
   Groq Llama model. A `verification` flag swaps in a skeptical,
   contradiction-checking system prompt; a `task_type` of `"label"` switches to
   the FEVER SUPPORTED/REFUTED/NEI verdict prompt.
5. **Evaluation** (`evaluator.py`) drives every dataset through the three
   experiments across the model / top-k sweep, scores the answers, optionally
   measures self-consistency, and writes the CSVs.

---

## ⚠️ Notes

- The first run downloads the MiniLM embedding model (~90 MB); afterwards it is
  cached and works offline.
- Answers are scored by simple, case-insensitive substring matching against the
  `correct_answer` / `poisoned_answer` strings in `questions.py`. Keep those
  strings short and distinctive (a year, a city) for reliable scoring.
- `temperature=0.0` is used so results are as reproducible as possible.
```
