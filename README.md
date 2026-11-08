# Name That LLM

Given one passage of text, guess which of 18 LLMs wrote it.

The training passages come from about 150 prompts, and every model answered every prompt once. The test set uses
new prompts, so the topic can't help: only writing style can. The model finds which passages answer the same
prompt, subtracts the shared topic, classifies the style, and uses "each model answers each prompt once" to fix its guesses.
The long explanation is in [WORKBOOK_EXPLAINED.md](WORKBOOK_EXPLAINED.md), and the walk-through is in
[notebooks/name_that_llm.ipynb](notebooks/name_that_llm.ipynb).

**Score** (balanced accuracy on prompts the model has never seen, 5-fold grouped CV, chance = 0.056):

| Setup | Score |
|---|---|
| TF-IDF style model | 0.477 |
| + 10% DistilBERT blend (hand-picked settings) | 0.521 |
| + Optuna-tuned settings, scored on the folds that picked them | 0.532 (too optimistic) |
| Hand-picked settings, honest nested check | 0.520 |
| Optuna-tuned settings, honest nested check | 0.520 |

**Tuning did not help.** The 0.532 comes from scoring on the same folds the search used. The honest check
(`python -m name_that_llm.tune --nested`: tune inside each fold, test on prompts the search never saw) gives 0.520
both ways: in four of five folds the search just kept the hand-picked settings. That check used 6 trials and 3 inner
folds, a lighter search than the 12-trial run, so a bigger search might still find something. Treat 0.52 as the
real level. All numbers are local estimates, not Kaggle scores.

## Notebook or package?

- **The training code lives in `src/name_that_llm/`.** The command line, the Airflow pipeline, the API and the tests
  all use it. Tuning and the final training run from there.
- **`notebooks/name_that_llm.ipynb` is a self-contained walk-through** of the method (clustering, features,
  centering, assignment, the DistilBERT blend). It has its own copy of the code and does not use the package or
  Optuna, so running it does not change the model the API serves.

## Two ways to work

You do **not** need Docker to develop or test. Docker is for running the API and the Airflow training pipeline.

### A. Everyday development (no Docker)

```bash
git clone https://github.com/sivaalli89/ML-Ops-Final.git
cd ML-Ops-Final
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt     # pinned versions
pip install -e ".[dev]"             # the package itself + pytest
pytest                               # about 5 seconds, uses made-up data
```

The competition data is not in git. Put `train.csv` and `test.csv` in `data/` (download them from the Kaggle
competition page; a shared DVC remote will replace this step). Then:

```bash
python -m name_that_llm.grouping          # finds the prompt groups -> data/train_with_groups.csv
python -m name_that_llm.tune --trials 15  # Optuna searches the settings -> artifacts/best_params.json
python -m name_that_llm.train --cv        # score on unseen prompts, about 1 minute (add --params artifacts/best_params.json)
python -m name_that_llm.train --fit       # trains on everything, saves artifacts/model.joblib
uvicorn api.main:app --port 8000          # serves the model
```

The DistilBERT numbers are read from `data/embeddings/distilbert.npy` if it exists. To compute them yourself,
`pip install sentence-transformers` (needs PyTorch); the notebook's section 7 shows how. Without the file, the
model trains and serves with the style features only.

### B. Run it in Docker

Two parts, both started from `docker-compose.yml`:

| Part | Command | What it does |
|---|---|---|
| **API + MLflow** | `docker compose up --build` | API on http://localhost:8000 (serves the registry's champion model, or `artifacts/model.joblib` if there is none yet); MLflow on http://localhost:5000 |
| **Training pipeline** | `docker compose --profile pipeline up --build` | Airflow on http://localhost:8080 (login `admin` / `admin`, change it in `.env`) |

The Airflow DAG `name_that_llm_training` runs, in order: check the data, find prompt groups, **tune with Optuna**,
train, evaluate, **register the model**. Switch it on and click *Trigger DAG w/ config* to choose the number of Optuna trials. It reads
`train.csv` and the embeddings from `./data` and writes the model, `best_params.json` and `metrics.json` to
`./artifacts`, which is the same folder the API reads. Restart the API (or just `docker compose restart api`) to
load a newly trained model.

`artifacts/model.joblib` is always the latest trained model. What the API serves is decided by the registry
(see *Model versioning*).

Give Docker about 8 GB of memory: tuning builds large feature tables. The images are slim and have no PyTorch; the
DistilBERT numbers come from the saved `data/embeddings/distilbert.npy`.

Without Docker you can do the same by hand with the commands in section A.

## Model versioning

Models are versioned in the **MLflow Model Registry** (http://localhost:5000 in Docker).

- Every registered model gets a **registry number** (1, 2, 3 ...) and a **semantic version tag** such as `0.1.2`.
  `X.Y` comes from `__version__` in `src/name_that_llm/__init__.py`: raise it when the features or model code change.
  `Z` goes up by one for every retrain under the same `X.Y`.
- The alias **`champion`** marks the version the API serves. A new version becomes the champion only if its
  cross-validated score is at least as good as the champion's (`--min-gain` demands a margin). Otherwise it is
  kept, tagged `not promoted`, and the old champion stays. A replaced champion is tagged `retired`.
- Each version links to its run, which holds the settings, the scores, `metrics.json` and `best_params.json`.
  Optuna's search is logged as its own run (one point per trial) when `MLFLOW_TRACKING_URI` is set.
- **A retrain, step by step:** train a candidate -> score it on unseen prompts -> register it as the next version
  with its semantic version and score -> compare with the champion -> move the `champion` alias only if it is not
  worse. The longer explanation, with an example and the rollback steps, is in
  [WORKBOOK_EXPLAINED.md](WORKBOOK_EXPLAINED.md#5-model-versioning-how-it-works).
- The API loads `models:/name_that_llm@champion` at startup. Set `MODEL_VERSION=2` to pin a version, which also
  works as a rollback. `GET /health` shows which version is loaded.

Without Docker, point MLflow at a local file and register by hand:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db          # Windows: set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python -m name_that_llm.train --cv --fit --metrics-out artifacts/metrics.json
python -m name_that_llm.registry                        # registers it and promotes it if it beats the champion
mlflow ui --backend-store-uri sqlite:///mlflow.db       # browse runs and versions
```

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
src/name_that_llm/   the package, one file per job:
    grouping.py          find prompt groups; centering and one-model-each assignment
    features.py          style features (TF-IDF) and DistilBERT embeddings
    model.py             the trained model object and how its two parts are blended
    train.py             train, cross-validate, save (python -m name_that_llm.train)
    tune.py              Optuna search (python -m name_that_llm.tune)
    registry.py          model versioning with MLflow (python -m name_that_llm.registry)
    predict.py           predict for a batch of passages
dags/                Airflow DAG: the training pipeline
docker/              Airflow image
api/                 FastAPI service
tests/               unit + end-to-end tests on toy data
notebooks/           the one explanatory notebook
Dockerfile, docker-compose.yml
```

## Status

| Part | State |
|---|---|
| Package, tuning, versioning, API, tests | Working. 36 tests pass. Tuning and registering have been run on the real data against a local MLflow store. |
| Docker images, the MLflow server and the Airflow pipeline | **Written but not yet run.** The compose file and DAG parse, but nobody has built the images or started the services. Expect to fix small problems on first run. |
| Monitoring and drift checks, CI | Not built yet. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
