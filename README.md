# Name That LLM

Given one passage of text, guess which of 18 LLMs wrote it.

The training passages come from about 150 prompts, and every model answered every prompt once. The test set uses new
prompts, so the topic can't help: only writing style can. So the notebook does this:

1. finds which passages answer the same prompt (clusters them by topic),
2. turns each passage into style numbers (character, word and "skeleton" TF-IDF),
3. subtracts each prompt group's average, which cancels the topic and keeps the style,
4. trains a classifier and tests it on prompts it has never seen,
5. uses "each model answers each prompt once" to clean up the guesses,
6. blends in a small share of a language model (DistilBERT) for a further gain.

Everything is in one notebook: [notebooks/name_that_llm.ipynb](notebooks/name_that_llm.ipynb). The longer explanation, with
the experiments that did and did not work, is in [WORKBOOK_EXPLAINED.md](WORKBOOK_EXPLAINED.md).

## Results

Balanced accuracy on prompts the model has never seen (5 folds split by prompt; guessing at random gives 0.056):

| Step | Score |
|---|---|
| Random split instead of by prompt (misleading), simple character-only model | 0.114 |
| Split by prompt, same simple model | 0.313 |
| + centering by prompt group | 0.381 |
| + one-model-each assignment | 0.422 |
| + skeleton features | 0.466 |
| Logistic regression tuned a little (C=1) | 0.477 |
| + 10% DistilBERT blend | **0.521** |

The first two rows use a simpler model, only to show why the split matters; the rest use the full features. The notebook does
not print the full model's score without centering (an earlier experiment gave 0.309).

These are local estimates on the training data, not scores from the Kaggle competition. The last row came out as 0.520 on
one machine and 0.521 on another: tiny numerical differences, so read it as "about 0.52".

## How to run it

The competition data is not in git. Put `train.csv` in `data/` first: either download it from the Kaggle competition page, or
run `dvc pull` once DVC storage is set up (see *Data and model versioning*). A full run needs a few GB of memory, because the
feature tables are large, and takes a while: about 6 to 7 minutes on your own Python, about 12 minutes in the Jupyter container
(measured before section 8 was added) and about 15 minutes under Airflow (measured here). See *Why a run takes a while* below.

### Option A: Python on your computer

```bash
git clone https://github.com/sivaalli89/ML-Ops-Final.git
cd ML-Ops-Final
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/name_that_llm.ipynb      # or open the notebook in VS Code
```

### Option B: Docker (JupyterLab in a container)

Same notebook, same library versions on every computer, nothing to install except Docker.

```bash
docker compose up --build
```

Then open http://localhost:8888/lab?token=namethatllm, open `notebooks/name_that_llm.ipynb` and run it top to bottom.
Stop it with `Ctrl+C` (or `docker compose down`).

- The notebook and `data/` are mounted from your computer, so your edits and the notebook's saved results stay in your
  project folder. Nothing is stored inside the container.
- The page is only reachable from your own computer, and needs the token. Change it with `JUPYTER_TOKEN` in a `.env` file
  (copy `.env.example`).
- To run the whole notebook without opening Jupyter, saving the result as `data/ran.ipynb` and leaving the original alone:
  `docker compose run --rm notebook jupyter nbconvert --to notebook --execute name_that_llm.ipynb --output-dir ../data --output ran`

### Option C: Airflow (run it on demand or on a schedule)

Without Airflow nothing starts the notebook: you press Run yourself. Airflow adds a trigger. It runs the notebook for you,
when you click a button or every week, keeps a log of every run, and saves a copy of the notebook with that run's results.

```bash
docker compose --profile airflow up --build airflow
```

Open http://localhost:8080 and log in with `admin` / `admin` (set `AIRFLOW_ADMIN_PASSWORD` in `.env` to change it; these defaults are for
your own computer, where the page is only reachable locally). The DAG
`run_notebook` is already switched on:

- **Run it now:** press the play button next to `run_notebook`, or `docker compose exec airflow airflow dags trigger run_notebook`.
- **Run it on a schedule:** in `dags/run_notebook_dag.py` change `schedule=None` to `"@weekly"` (or `"@daily"`).
- **What it does:** three tasks in order: `check_data` (is `train.csv` there?), `run_notebook`, `show_model_info`.
- **Where the results go:** each run saves its own executed copy as `runs/run_<time>.ipynb` (your original notebook is never
  changed), and the trained model and its info go to `models/`.

`dags/run_notebook_dag.py` is the only Python file in the project. It has no model code, it only runs the notebook, so the
notebook stays the one place where the work happens. This is a single small container with a SQLite database, which is
plenty for running one notebook.

**DistilBERT numbers (section 7).** The notebook reads `data/embeddings/distilbert.npy` if it exists. That file is not in git,
so on a fresh copy the notebook computes it once, which needs `sentence-transformers` (it installs PyTorch) and takes a few
minutes on a CPU. With Option A, `pip install sentence-transformers`. With Option B, build with
`INSTALL_EMBEDDINGS=true docker compose up --build` (PowerShell: `$env:INSTALL_EMBEDDINGS="true"; docker compose up --build`),
which makes the image about 1 GB bigger. After the first run the file is reused.

## Where it trains, and what "deployed" means here

Training happens inside the notebook, on whichever Python runs it: your virtual environment (Option A), the Jupyter
container (Option B) or the Airflow container (Option C), the last two with Python 3.12 and pinned libraries.

Section 8 of the notebook trains a final model on all of `train.csv` and saves it as a numbered version in `models/`.
**Nothing serves predictions**: there is no web service, so the model is saved and versioned but not deployed. (The archived
pipeline version, below, had one.) If you need predictions for the competition, the next step would be a cell that loads
the saved model and writes a prediction for each passage in `test.csv`.

## Data and model versioning

Both are versioned with **DVC** plus **git tags**. DVC stores a large file by its hash and gives git a tiny pointer file
(`.dvc`), so git history records exactly which data and which model went together, without containing them.

| What | Pointer in git | Tag | Made by |
|---|---|---|---|
| Training data (`train.csv`, `test.csv`, the DistilBERT numbers) | `data/*.dvc` | `data-v1`, `data-v2` ... | you, when the data changes |
| Trained model | `models/model.joblib.dvc` and `models/model_info.json` | `model-v0.1.0` ... | notebook section 8, then the commands below |

`models/model_info.json` says which version it is, its cross-validated score, the settings, and the hash of the `train.csv` it was
trained on, so every model names its data.

**Install DVC** (not part of `requirements.txt`, since the notebook doesn't need it): `pip install dvc`.

**Get the data** (instead of downloading it by hand), once a DVC storage is set up: `dvc pull`.

**Save a new model version** after the notebook has run:

```bash
dvc add models/model.joblib
git add models/model.joblib.dvc models/model_info.json models/.gitignore
git commit -m "Model v0.1.0"
git tag model-v0.1.0
dvc push
```

**Go back to an older version:** `git checkout model-v0.1.0 -- models/model.joblib.dvc models/model_info.json`, then
`dvc checkout models/model.joblib`. The same works for data with a `data-v1` tag and the files in `data/`.

**Version numbers.** Raise `MODEL_VERSION` in notebook section 8 when the method changes (new features, a different
classifier). A plain retrain on the same method can raise the last digit.

**Where DVC keeps the files.** On this computer the storage is the folder `D:/dvc-storage/ml-ops-final`, set in
`.dvc/config.local`, which git ignores because the path belongs to one machine. **Teammates can't reach that folder.** For the
team, point DVC at storage everyone can use (Google Drive, S3, Azure Blob, an SSH server or a shared network folder), for
example `dvc remote add -d storage <address>`, then commit the change to `.dvc/config`. Copying `data-v1` to the shared
storage with `dvc push` makes `dvc pull` work for everyone. Check the competition's rules before sharing its data.

**Limits.** This tracks versions and lets you restore them. It doesn't compare a new model against the old one or decide which
one is "live"; for that kind of model registry see the archived version.

## Why a run takes a while

The notebook trains about 30 large models, one after another, and each one works on a table of roughly 55,000 columns (the
character, word and skeleton features) for 2,700 passages:

| Where | Models trained |
|---|---|
| Section 3, random versus split-by-prompt scoring | 10 |
| Section 6, with and without the skeleton features | 10 |
| Section 7, the DistilBERT blend | 10 (5 TF-IDF + 5 DistilBERT) |
| Section 8, the final model | 2 |

Sections 6 to 8 train one model after another, and each fit uses only a few cores (we saw 1.5 to 4.5 of 16 busy in the Airflow
container), so most of the machine sits idle. Building the feature tables adds time on top.
Containers are slower than your own Python: they share the machine with Docker, and under Airflow the scheduler and web page
run alongside. The notebook is written as a plain loop so it is easy to read. If speed matters more than readability, the
folds could run in parallel, which would cut the time to a few minutes; the archived pipeline version did that.

## What has been tested

| Part | Tested how |
|---|---|
| The notebook | run top to bottom with no errors on your own Python (Windows, Python 3.13, 6.4 minutes), in the Jupyter container and under Airflow; the final version, including the model-saving section, was run natively and under Airflow, with the same scores |
| Airflow | the `run_notebook` DAG was triggered by hand and finished with state `success` (about 15 minutes); the executed copy, the model and its info were written |
| Data versioning | files pushed to DVC storage, deleted together with the local cache, and restored by `dvc pull` with identical hashes |
| Model versioning | a model saved as `model-v0.1.0`, restored in a fresh clone with `dvc pull`, and rolled back and forward between two versions |
| The saved model | loads in plain Python with only scikit-learn installed |
| The Jupyter container | starts, needs the token, and can write to `data/` and `models/` (which section 8 needs) |

Not tested: building the image with `INSTALL_EMBEDDINGS=true` (the PyTorch route), running on macOS or Linux (everything here ran
on Windows with Docker Desktop; on Linux the file ownership of the mounted folders may need `UID` set), a **scheduled** run (only a
manual trigger was run), DVC storage shared between several people (the storage here is a
folder on one computer), and Airflow with a bigger setup than one container. The second model version in the rollback test was made
by editing the saved model's blend weight, so it tested the versioning mechanics, not a real retrain.

## What is in the folder

```
notebooks/name_that_llm.ipynb   the whole project, with results saved in it
dags/run_notebook_dag.py        the Airflow trigger that runs the notebook (Option C)
models/                         the saved model and its info (model.joblib is kept by DVC, not git)
runs/                           executed copies of the notebook, one per Airflow run (not in git)
WORKBOOK_EXPLAINED.md           the method and the experiments in more detail
data/                           train.csv goes here; kept by DVC, git holds only the .dvc pointer files
.dvc/                           DVC settings
figures/                        charts used by WORKBOOK_EXPLAINED.md
research/                       the LLMmap paper (USENIX Security 2025)
requirements.txt                the libraries to install
Dockerfile, docker-compose.yml, docker/   run the notebook in JupyterLab (B) or under Airflow (C) in containers
CONTRIBUTING.md                 how to work on it as a team
```

## The earlier pipeline version

An earlier version also had a `.py` package, a FastAPI service, a larger Airflow pipeline (group, tune, train, evaluate,
register), an MLflow model registry that promoted a model only if it scored at least as well as the current one, Optuna tuning,
tests and Docker files for them. The Airflow DAG now in the project is much smaller: it only runs the notebook. It was removed to keep this project to the notebook, but nothing is lost:
it is saved under the git tag `archive/pipeline-version`. If the tag isn't on your copy, the same version is the commit
`dfdf20d`, which is in this repository's history (`git checkout dfdf20d`).

```bash
git checkout archive/pipeline-version        # look at it (git checkout - to come back)
git push origin archive/pipeline-version     # put the tag on GitHub so it is not only on this computer
```

Only the Python side of that version was ever tested. Its Docker images, MLflow server and Airflow pipeline were written but
never built or run.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
