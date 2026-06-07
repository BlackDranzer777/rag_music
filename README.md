# 🎵 RAG Misinformation Robustness Tester

A small Retrieval-Augmented Generation (RAG) project for a university NLP
course. It answers questions about music artists from a local knowledge base,
then tests **how easily the system is fooled by injected fake ("poisoned")
documents** — and whether a *verification prompt* makes it more robust.

Everything runs on **free** tools:

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
│   ├── real/          # real artist .txt files (1 example provided)
│   └── poisoned/      # fake .txt files (1 example provided)
├── src/
│   ├── ingestion.py   # load .txt, embed locally, store in ChromaDB
│   ├── retriever.py   # question -> top-k relevant chunks
│   ├── pipeline.py    # question + chunks -> Groq LLM -> answer (+verify toggle)
│   └── evaluator.py   # runs the 3 experiments, saves CSV
├── notebooks/
│   └── demo.ipynb     # loads results CSV, draws simple charts
├── results/           # experiment_results.csv lands here
├── questions.py       # evaluation question set
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
- toggle **"Include poisoned documents"** to add/remove the fake sources,
- toggle **"Verification mode"** to enable the skeptical, contradiction-checking
  prompt,
- ask a question and inspect both the answer and the retrieved sources
  (poisoned sources are flagged ☠️).

### The experiments (generates the CSV)

```bash
python -m src.evaluator
```

This runs all three experiments over every question in `questions.py` and
writes `results/experiment_results.csv`.

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

**Metrics** (printed and saved per experiment):

- **accuracy** — fraction of questions where the *correct* answer appears in the
  model's response.
- **hallucination_rate** — fraction of poisoned-target questions where the model
  repeated the *fake* answer. This is the core "did the misinformation win?"
  number.

The expected story: hallucination_rate jumps from Exp 1 → Exp 2, then drops
again in Exp 3 if verification helps.

---

## 🧠 How it works (quick tour for the report)

1. **Ingestion** (`ingestion.py`) reads every `.txt`, splits it into overlapping
   chunks, embeds each chunk locally with MiniLM, and stores the vectors in a
   ChromaDB collection. Real vs poisoned origin is saved in each chunk's
   metadata.
2. **Retrieval** (`retriever.py`) embeds the question and returns the top-k most
   similar chunks by cosine similarity.
3. **Generation** (`pipeline.py`) stuffs those chunks into a prompt and calls the
   Groq Llama model. A `verification` flag swaps in a system prompt that tells
   the model to check sources for contradictions before answering.
4. **Evaluation** (`evaluator.py`) wires the above together, runs the three
   experiments, scores the answers, and writes the CSV.

---

## ⚠️ Notes

- The first run downloads the MiniLM embedding model (~90 MB); afterwards it is
  cached and works offline.
- Answers are scored by simple, case-insensitive substring matching against the
  `correct_answer` / `poisoned_answer` strings in `questions.py`. Keep those
  strings short and distinctive (a year, a city) for reliable scoring.
- `temperature=0.0` is used so results are as reproducible as possible.
```
