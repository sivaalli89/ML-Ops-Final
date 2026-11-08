# Name That LLM

Given one passage of text, guess which of 18 LLMs wrote it.

The training passages come from about 150 prompts, and every model answered every prompt once. The test set uses
new prompts, so the topic can't help: only writing style can. The model finds which passages answer the same
prompt, subtracts the shared topic, classifies the style, and uses "each model answers each prompt once" to fix its guesses.
The long explanation is in [WORKBOOK_EXPLAINED.md](WORKBOOK_EXPLAINED.md), and the walk-through is in
[notebooks/name_that_llm.ipynb](notebooks/name_that_llm.ipynb).

**Score** (balanced accuracy on prompts the model has never seen, 5-fold grouped CV, chance = 0.056):
TF-IDF style model 0.477, with a 10% DistilBERT blend 0.521. This is a local estimate, not a Kaggle score.

## Two ways to work

You do **not** need Docker to develop or test. Docker is only for running the API and, later, the full stack.

### A. Everyday development (no Docker)

```bash
git clone https://github.com/sivaalli89/ML-Ops-Final.git
cd ML-Ops-Final
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
pytest                               # about 5 seconds, uses made-up data
```

The competition data is not in git. Put `train.csv` and `test.csv` in `data/` (download them from the Kaggle
competition page; a shared DVC remote will replace this step). Then:

```bash
python -m name_that_llm.grouping          # finds the prompt groups -> data/train_with_groups.csv
python -m name_that_llm.train --cv        # honest score, about 1 minute
python -m name_that_llm.train --fit       # trains on everything, saves artifacts/model.joblib
uvicorn api.main:app --port 8000          # serves the model
```

The DistilBERT numbers are read from `data/embeddings/distilbert.npy` if it exists. To compute them yourself,
`pip install sentence-transformers` (needs PyTorch); the notebook's section 7 shows how. Without the file, the
model trains and serves with the style features only.

### B. Run the API in Docker

```bash
python -m name_that_llm.train --fit       # creates artifacts/model.joblib first
docker compose up --build
curl localhost:8000/health
```

The image is the slim one (TF-IDF style model, no PyTorch). The model file is mounted in, not baked in.

## Using the API

`POST /predict` takes a **batch** of passages, not a single one:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"passages": ["first passage ...", "second passage ..."]}'
```

Centering and assignment need the other passages written for the same prompts, so the service clusters the batch
first. A batch under 54 passages can't be grouped; it is answered by the plain classifier and the response says so
(`mode: "ungrouped"`, plus a `note`). For real use send as many passages as you have.

## Project layout

```
src/name_that_llm/   grouping, group_ops, features, model, train, predict
api/                 FastAPI service
tests/               unit + end-to-end tests on toy data
notebooks/           the one explanatory notebook
Dockerfile, docker-compose.yml
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
