# Name That LLM

Given one passage of text, guess which of 18 LLMs wrote it.

The training passages come from about 150 prompts, and every model answered every prompt once. The test set uses new
prompts, so the topic can't help: only writing style can. So the notebook does this:

1. finds which passages answer the same prompt (clusters them by topic),
2. turns each passage into style numbers (character, word and "skeleton" TF-IDF),
3. subtracts each prompt group's average, which cancels the topic and keeps the style,
4. trains a classifier and tests it on prompts it has never seen,
5. uses "each model answers each prompt once" to clean up the guesses,
6. blends in a small share of a language model (DistilBERT) for a further gain,
7. holds out 20% of the prompts, lets **Optuna** choose the settings on the rest, trains the final model and scores it on the
   held-out prompts.

The method lives in one notebook: [notebooks/name_that_llm.ipynb](notebooks/name_that_llm.ipynb). Around it is a small MLOps
system: **Airflow** runs the notebook, **MLflow** tracks the run and keeps the model registry, a **FastAPI** service serves the
champion model and **monitors** what it is sent for drift. The longer explanation of the method, with the experiments that
did and did not work, is in [WORKBOOK_EXPLAINED.md](WORKBOOK_EXPLAINED.md).

```
 data/train.csv ──▶ Airflow: train_pipeline ─────────────────────────────────────────────┐
  (DVC)              1 check_data                                                        │
                     2 run_notebook   notebook: group, features, Optuna, train, score    │
                         └──▶ models/  model.joblib, model_info.json, holdout.csv,        │
                                       optuna_trials.csv         (+ runs/ executed copy)  │
                     3 register_model ──▶ MLflow: run, metrics, Optuna trials, files ◀───┘
                                              │   new version 0.2.x; "champion" moves to it only if
                                              ▼   its held-out score is at least as good
                                       model registry ──▶ API (FastAPI) loads the champion by itself
                                                              POST /predict   GET /monitor   GET /dashboard
                                                                   │
                       pipeline/simulate_drift.py ────────────────┘  clean data, then corrupted data
```

## How this meets the course project outline

| Requirement | Where it is |
|---|---|
| Dataset with a clear target, a metric | the Kaggle "Name That LLM" data; the 18 LLMs are the target; balanced accuracy (see *Results* for what it can and can't say) |
| Train/test split, test set isolated | 20% of the prompts are held out (notebook section 8) and never used for training or tuning; `models/holdout.csv` |
| Orchestrator (Airflow) | `dags/train_pipeline_dag.py`: `check_data`, `run_notebook`, `register_model` |
| Experiment tracking (MLflow) | every run logs its settings, scores, Optuna trials and files (`pipeline/register_model.py`) |
| Hyperparameter search | **Optuna** searches three settings in notebook section 8 |
| Model registry, semantic versioning | MLflow registry: versions `0.2.0`, `0.2.1` ..., and a `champion` alias that is only moved to a model that scores at least as well |
| Container + deployed inference | Docker + FastAPI (`service/api.py`) loads the champion and returns predictions |
| Monitoring dashboard | Evidently 0.7.23 data drift reports: `/dashboard`, `/dashboard/evidently` and `/monitor` (JSON) |
| Baseline validation, drift simulation, anomaly verification | `pipeline/simulate_drift.py`; results in *Drift test* below |
| Repository, README, dependencies | this file, `requirements.txt`, the Dockerfiles in `docker/` |

## Results

Balanced accuracy on prompts the model has never seen (guessing at random gives 0.056). The first rows are 5-fold
cross-validation over all prompts, split by prompt:

| Step | Score |
|---|---|
| Random split instead of by prompt (misleading), simple character-only model | 0.114 |
| Split by prompt, same simple model | 0.313 |
| + centering by prompt group | 0.381 |
| + one-model-each assignment | 0.422 |
| + skeleton features | 0.466 |
| Logistic regression tuned a little (C=1) | 0.477 |
| + 10% DistilBERT blend (settings picked by hand on these folds) | **0.521** |

The first two rows use a simpler model, only to show why the split matters; the rest use the full features. The notebook does
not print the full model's score without centering (an earlier experiment gave 0.309).

The model that is actually deployed is scored differently, because the settings are now chosen by Optuna and the model is
trained on 80% of the prompts:

| The deployed model | Score |
|---|---|
| Optuna's cross-validated score on the 80% training prompts (slightly optimistic: best of 20 tries) | 0.517 |
| Same folds, hand-picked settings (C=1, C=0.1, weight 0.1) | 0.517 |
| **Held-out prompts** (20% of the prompts, 30 prompts, never used for anything) | **0.478** |

Read these with care. The held-out score comes from only about 30 prompts, so it is noisy by a few points, and
the model behind it saw 20% less data than the 0.521 row, so the two numbers are not directly comparable. Optuna was started
from the hand-picked settings, and none of its 20 trials beat them: it ended on the same settings, a gain of 0.000. So Optuna is in the pipeline and works, but on this data it confirmed the hand-picked settings instead of improving them.

These are local estimates on the training data, not scores from the Kaggle competition, and balanced accuracy is my choice of
metric: check it against the competition's evaluation page. Scores can differ by about 0.001 between machines (tiny
numerical differences), so read them as "about".

## How to run it

The competition data is not in git. Put `train.csv` in `data/` first: either download it from the Kaggle competition page, or
run `dvc pull` once DVC storage is set up (see *Data and model versioning*). A full run needs a few GB of memory, because the
feature tables are large, and it takes a while (see *Why a run takes a while*).

### Option A: Python on your computer (the notebook only)

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

Either way the notebook writes `models/model.joblib`, `model_info.json`, `holdout.csv` and `optuna_trials.csv`. That trains a
model but does not register it or serve it; for that, use Option C.

### Option C: the whole system (Airflow trains, MLflow registers, the API serves and monitors)

```bash
docker compose --profile airflow --profile api up --build -d airflow api      # MLflow starts too
```

| What | Address | Login |
|---|---|---|
| Airflow (the trigger) | http://localhost:8080 | `admin` / `admin` (`AIRFLOW_ADMIN_PASSWORD` in `.env` changes it) |
| MLflow (runs and the model registry) | http://localhost:5050 (`MLFLOW_PORT` changes it) | none |
| The API (try it) | http://localhost:8000/docs | none |
| The monitor | http://localhost:8000/dashboard | none |

These defaults are for your own computer: every page is only reachable from it.

1. **Train.** In Airflow, press the play button next to `train_pipeline` (or
   `docker compose exec airflow airflow dags trigger train_pipeline`). Three tasks run in order: `check_data` (is `train.csv`
   there?), `run_notebook` (tunes with Optuna, trains, scores, saves the files), `register_model` (logs to MLflow, registers a
   version, promotes it if it is good enough). It takes about 26 minutes. Each run keeps its own executed copy of the
   notebook in `runs/`; your original is never changed.
2. **Look at the run.** In MLflow, the experiment `name-that-llm` has one run per training (parameters, metrics, one nested run
   per Optuna trial, the files), and *Models* has the registry.
3. **The API picks the champion up by itself.** Until the first model is registered, `/predict` answers 503 ("no model yet").
   Within about 10 seconds of a new champion, the next request is answered by it. `GET /model` says which version and where it came from.
4. **Send passages** to `POST /predict` (below).
5. **Run the drift test** (below).

The schedule is switched off. To train every week, change `schedule=None` to `"@weekly"` in `dags/train_pipeline_dag.py`.
Stop everything with `docker compose --profile airflow --profile api down` (add `-v` to also delete the MLflow and Airflow data).

**DistilBERT numbers (notebook section 7).** The notebook reads `data/embeddings/distilbert.npy` if it exists. That file is not in git,
so on a fresh copy the notebook computes it once, which needs `sentence-transformers` (it installs PyTorch) and takes a few
minutes on a CPU. With Option A, `pip install sentence-transformers`. With Options B and C, build with
`INSTALL_EMBEDDINGS=true` (PowerShell: `$env:INSTALL_EMBEDDINGS="true"; docker compose up --build`), which makes the image
about 1 GB bigger. After the first run the file is reused.

## The prediction service

`service/api.py` loads the **champion** model from the MLflow registry and answers requests. (Without `MLFLOW_TRACKING_URI`, for example
`uvicorn service.api:app --port 8000` from the project folder, on your own Python, it loads the files in `models/` instead and notices when `model_info.json`
changes.)

| Endpoint | What it does |
|---|---|
| `POST /predict` | body `{"passages": ["...", "..."]}`, the passages that answer **one** prompt (2 to 200); returns `{"model_version", "used_embeddings", "llms": [...]}`, one guess per passage, in order |
| `GET /model` | which model is loaded: its info (version, held-out score, settings, training-data hash), where it came from, and its accuracy as served on the held-out prompts |
| `GET /monitor` | the drift report as JSON (below) |
| `GET /dashboard` | the same report as a web page that refreshes itself; `/dashboard/evidently` is Evidently's own full report |
| `GET /health` | `{"status": "ok", "model_loaded": true}` |

Example, from PowerShell:

```powershell
$body = @{ passages = @("First answer to the prompt...", "Second answer...", "Third answer...") } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $body
```

- **Send one prompt's passages together.** Centering and the one-answer-per-LLM step need the other passages from the same
  prompt (the notebook's last point). One passage can't be centered, so a request with fewer than 2 is refused (HTTP 422). With
  fewer than 18 passages each one still gets a different LLM; with more than 18, an LLM can repeat. The model works best with
  all 18 answers to a prompt; `test.csv` has no `prompt_id`, so real use needs the grouping step (notebook section 2) first.
- **New models load by themselves.** Every 10 seconds at most, a request asks the registry whether the champion changed. If it
  did (a new model was promoted, or someone moved the alias back), the service downloads and loads it. If that fails, it keeps
  answering with the model it has. No restart, no rebuild.
- **DistilBERT is optional here.** The 10 to 25% DistilBERT part needs `sentence-transformers` (PyTorch, about 1 GB, and the
  language model is downloaded at the first start). Without it the service answers with the style model only, which is a
  little less accurate, and `used_embeddings` says `false`. Build with `INSTALL_EMBEDDINGS=true` to include it.
- The service has no login. It is only reachable from your own computer (the port is bound to 127.0.0.1).

## Experiment tracking and the model registry

`pipeline/register_model.py` runs as the last Airflow task. It reads what the notebook saved and:

1. **Logs a run** in the MLflow experiment `name-that-llm`: the settings, training-data hash and passage counts as parameters;
   the held-out score and the cross-validated score as metrics; one nested run per Optuna trial; and the model, `model_info.json`,
   `holdout.csv`, `optuna_trials.csv` and the executed notebook as artifacts.
2. **Registers** the model files as a new version of the registered model `name-that-llm`.
3. **Gives it a semantic version.** The notebook's `MODEL_VERSION` (now `0.2.0`) is used if no registered version has it yet;
   otherwise the last number goes up: `0.2.1`, `0.2.2` ... Raise `MODEL_VERSION` in the notebook when the method changes (new
   features, another kind of classifier) and the registry starts counting from the new number. The version number is stored as a
   tag on the registry version and inside the `model_info.json` that travels with it.
4. **Promotes it, or not.** The alias `champion` moves to the new version only if its held-out score is at least as good as
   the current champion's. A worse model is still registered, so it can be looked at, but the old champion keeps serving.

To roll back, move the alias to an older version in the MLflow page (*Models*, the version, *Aliases*) or with
`MlflowClient().set_registered_model_alias("name-that-llm", "champion", "<version>")`; the API loads it within seconds.

Optuna searches three settings, on the 80% training prompts only: `style_c`, `embedding_c` and `embedding_weight`. Fitting is
the slow part, so the two C settings take three values each and each value is fitted once per fold; Optuna then picks the best
combination from those results. Its first trial is the hand-picked setting. In earlier experiments tuning did not help once it was scored honestly, and
that holds up here (see *Results*).

## Monitoring and drift

In production nobody tells the service whether its guesses were right, so it watches what it can see: the passages it is sent
and how confident the model is. For every passage `service/monitoring.py` records eight numbers:

| Number | Why |
|---|---|
| passage length (characters, words) | truncated or padded input |
| average word length, share of punctuation and symbols | garbled or machine-generated text |
| share of words not seen in training | corrupted text, another language, another domain |
| share of neighbouring word pairs seen in training | the words are right but their order is not (shuffled text) |
| similarity to the other passages in the request | passages from different prompts mixed in one request (the grouping the model relies on is broken) |
| the model's top probability for the passage | the model is unsure of what it sees |

**Baseline.** When a model is loaded, the service runs half of the held-out prompts (the even-numbered ones) through its own code
and keeps the eight numbers. The other half is kept as clean test data for the drift test. (If the test used the baseline
passages themselves it could not fail.)

**Check, with Evidently.** The monitor is [Evidently](https://www.evidentlyai.com/) 0.7.23. Its data drift report
(`DataDriftPreset`) compares the recent passages (the last 500 by default) with the baseline, one column at a time. I use its
**normalised Wasserstein distance**: how far the recent values have moved, in units of the baseline's own spread (0 = same).
A column has **drifted** when that distance is above **1.0**. The status is:

- `OK`: nothing drifted
- `ALERT`: at least one column drifted, or 3 or more of the last 100 requests were rejected as malformed (wrong field names,
  wrong types, too few passages): a schema change
- `WAITING`: fewer than 80 recent passages (about 5 prompts), too few to tell

**Why a distance and not Evidently's default p-value test.** The first version used a Kolmogorov-Smirnov test (p < 0.001).
Run on 10 random clean samples of 5 held-out prompts it raised a false alarm on **6 of 10**. The reason: the passages of one
prompt resemble each other, so 90 passages from 5 prompts are not 90 independent rows, and p-value tests are over-confident
about them. A distance measures how big the move is, whatever the sample size. The threshold was then chosen from measurements
on this model's held-out prompts (normalised distance of the worst column):

| Sample | Largest distance seen |
|---|---|
| clean, 3 prompts (20 random samples) | 1.35 |
| clean, 5 prompts | 0.84 |
| clean, 8 prompts | 0.54 |
| clean, 12 prompts | 0.33 |
| corrupted, 5 prompts: the weakest signal (`truncated`) | 1.65 |

Hence the threshold of 1.0 and the 80-passage minimum (3-prompt samples can exceed 1.0 by chance). With these settings, 10
random clean samples of 5 prompts raised **0 of 10** false alarms. Be careful with that evidence: the threshold was picked from the same held-out
prompts and the same corruptions the drift test uses, so the clean-versus-corrupted gap is probably larger here than it would be on
new data or other kinds of drift. The eighth number (word pairs) was added after the first drift test showed that shuffled
words slipped past the other seven.

Open http://localhost:8000/dashboard to see it (a compact table that refreshes itself), or
http://localhost:8000/dashboard/evidently for Evidently's own report with a distribution plot per column. `GET /monitor` gives
the same as JSON, for a script to poll. Requests and their numbers are also written to `monitoring/requests.jsonl`, so the
history survives a restart. What it can't do: it can't see wrong answers (there are no labels in production), it has no
e-mail or chat alert (the status is something you look at or poll), and requests with only a few passages look different from the
baseline, which is made of whole prompts, so they can raise an alert on purpose.

## Drift test

`pipeline/simulate_drift.py` is the stress test. With the system from Option C running:

```bash
docker compose --profile api --profile drift run --rm drift-test
```

It sends the clean half of the held-out prompts, one prompt per request, and compares the accuracy with what the model
scored when it was trained and the monitor with the baseline. Then it corrupts the same passages and sends them again:

| Corruption | What is done |
|---|---|
| `truncated` | only the first fifth of every passage |
| `random_chars` | 30% of the letters replaced by random ones |
| `shuffled_words` | the words inside every passage shuffled |
| `mixed_prompts` | passages from different prompts in the same request |
| `broken_schema` | requests with the wrong field names, wrong types or too few passages |

Results of the run documented here (model 0.2.0, the clean half of the held-out prompts):

| Scenario | Accuracy | Monitor | What the monitor saw |
|---|---|---|---|
| clean (12 prompts, 216 passages) | 0.440 | OK | nothing unusual |
| `truncated` | 0.227 | **ALERT** | passage length (characters and words) |
| `random_chars` | 0.296 | **ALERT** | unknown words, known word pairs, similarity to the other passages |
| `shuffled_words` | 0.380 | **ALERT** | known word pairs |
| `mixed_prompts` | 0.315 | **ALERT** | similarity to the other passages |
| `broken_schema` | n/a | **ALERT** | 6 requests rejected with HTTP 422 |

Accuracy here is plain accuracy as the service gives it: the style model only, because the API image has no DistilBERT, so
the clean figure (0.440) is below the notebook's 0.478. The point of the column is the fall under corruption (0.44 to between
0.23 and 0.38), which the monitor reports without being told the answers. 10 random clean samples of 5 prompts, sent through
the same stack, raised 0 false alarms. Every corruption is caught by the numbers built to catch it, and the weakest case is
`shuffled_words`, caught by one number only (known word pairs), which was added after the first version missed it.

The accuracy column is possible only because the script knows the true answers; the live monitor has to work from the inputs.
The raw results are saved in `monitoring/drift_test.json`.

## Data and model versioning

Both are versioned with **DVC** plus **git tags**. DVC stores a large file by its hash and gives git a tiny pointer file
(`.dvc`), so git history records exactly which data and which model went together, without containing them. This sits next
to the MLflow registry: DVC and git say *which files existed when*; the registry says *which model is serving*.

| What | Pointer in git | Tag | Made by |
|---|---|---|---|
| Training data (`train.csv`, `test.csv`, the DistilBERT numbers) | `data/*.dvc` | `data-v1`, `data-v2` ... | you, when the data changes |
| Trained model | `models/model.joblib.dvc` and `models/model_info.json` | `model-v0.2.0` ... | notebook section 9, then the commands below |

`models/model_info.json` says which version it is, its held-out score, the settings, and the hash of the `train.csv` it was
trained on, so every model names its data. `models/holdout.csv` is rebuilt by the notebook and is not versioned.

**Install DVC** (not part of `requirements.txt`, since the notebook doesn't need it): `pip install dvc`.

**Get the data** (instead of downloading it by hand), once a DVC storage is set up: `dvc pull`.

**Save a new model version** after the notebook has run:

```bash
dvc add models/model.joblib
git add models/model.joblib.dvc models/model_info.json models/optuna_trials.csv models/.gitignore
git commit -m "Model v0.2.0"
git tag model-v0.2.0
dvc push
```

**Go back to an older version:** `git checkout model-v0.1.0 -- models/model.joblib.dvc models/model_info.json`, then
`dvc checkout models/model.joblib`. The same works for data with a `data-v1` tag and the files in `data/`. (The tags exist only
on the computer that made them until someone runs `git push origin --tags`.)

**Where DVC keeps the files.** On this computer the storage is the folder `D:/dvc-storage/ml-ops-final`, set in
`.dvc/config.local`, which git ignores because the path belongs to one machine. **Teammates can't reach that folder.** For the
team, point DVC at storage everyone can use (Google Drive, S3, Azure Blob, an SSH server or a shared network folder), for
example `dvc remote add -d storage <address>`, then commit the change to `.dvc/config`. Copying `data-v1` to the shared
storage with `dvc push` makes `dvc pull` work for everyone. Check the competition's rules before sharing its data.

## Why a run takes a while

The notebook fits about 60 models one after another, about 40 of them on tables of roughly 55,000 columns (the character, word
and skeleton features) for 2,000 to 2,700 passages:

| Where | Models fitted |
|---|---|
| Section 3, random versus split-by-prompt scoring | 10 |
| Section 6, with and without the skeleton features | 10 |
| Section 7, the DistilBERT blend | 10 (5 TF-IDF + 5 DistilBERT) |
| Section 8, every candidate C once per fold | 30 (15 TF-IDF + 15 DistilBERT); Optuna's 20 trials then reuse them |
| Section 9, the final model | 2 |

Each fit uses only a few cores (we saw 1.5 to 4.5 of 16 busy in the Airflow container, more in other places), so much of the
machine can sit idle. Building the feature tables adds time on top. Containers are slower than your own Python: they share
the machine with Docker, and under Airflow the scheduler, web page and MLflow run alongside. The notebook is written as a plain
loop so it is easy to read. If speed matters more than readability, the folds could run in parallel; the archived pipeline
version did that.

## What has been tested

| Part | Tested how |
|---|---|
| The notebook | run top to bottom with no errors in the Airflow container (the `run_notebook` task took 25.9 minutes); the saved copy in `notebooks/` is that run. An earlier version of the same sections (before the Optuna starting trial and the file permissions fix) also ran in the Jupyter container. It has not been run on a plain local Python since the Optuna and held-out sections were added |
| Airflow | `train_pipeline` triggered by hand: all three tasks succeeded, 26 minutes in total. Two earlier runs failed: one on a file permission error (fixed in the notebook), and one because the containers were stopped from outside while it ran |
| MLflow | after the run: one registered model with version `0.2.0` as champion, 21 runs (the training run and its 20 Optuna trials), and the model, `model_info.json`, held-out passages, trials and the executed notebook as artifacts |
| Registry rules | on the live stack: registering the same files again gave `0.2.1`, promoted, and the API switched to it within 4 seconds; registering a copy with its score set to 0.40 by hand gave `0.2.2`, registered but not promoted, and the API kept serving `0.2.1`. Moving the alias back made the API follow. These test entries were deleted afterwards. The same rules were first tested with dummy files |
| The API | starts with no model and answers 503; picks the first champion up by itself; starts with a champion already registered; `/predict`, `/model`, `/monitor`, `/dashboard` and `/dashboard/evidently` answered; malformed requests get 422 |
| Monitoring and drift | the drift test above on the live stack; 0 false alarms in 10 clean samples of 5 prompts, on the final model and on an earlier one |
| The Docker images | built and run: notebook, Airflow, MLflow, API and the drift test |
| The saved model | loads in plain Python with only scikit-learn installed (checked for the first version of the model file; the format has not changed) |

Not tested: the API with DistilBERT (the PyTorch route, `INSTALL_EMBEDDINGS=true`, for both the notebook and the API), so the
service has only run in style-only mode (0.440 against 0.478 for the full ensemble); a **scheduled** run (only manual triggers);
macOS or Linux (everything ran on Windows with Docker Desktop; on Linux the file ownership of the mounted folders may need `UID`
set); DVC storage shared between several people (the storage here is a folder on one computer); a retrain that really
produces a better or worse model (the promotion rules were exercised with the same model files and a hand-edited score, and
this retrain only reproduced the hand-picked settings); a test of the drift test on data other than this held-out set. There
are no unit tests (the archived version had 38).

**One file is out of date on purpose:** `models/model.joblib.dvc` still points at the first model (0.1.0, trained on all
passages). The new `model.joblib` has not been added to DVC; run the commands in *Data and model versioning* to version it
(`dvc add models/model.joblib`, then commit and tag `model-v0.2.0`).

## What is in the folder

```
notebooks/name_that_llm.ipynb   the method and the training, with results saved in it
dags/train_pipeline_dag.py      the Airflow DAG: check data, run the notebook, register the model
pipeline/register_model.py      logs a run to MLflow, registers the model, promotes it if it is good enough
pipeline/simulate_drift.py      the drift test (clean data, then corrupted data) against the running API
service/                        api.py (the FastAPI service) and monitoring.py (its drift monitoring)
models/                         the saved model, its info, the held-out passages, Optuna's trials (model.joblib is kept by DVC, not git)
runs/                           executed copies of the notebook, one per Airflow run (not in git)
monitoring/                     only the API's request log and the drift test's results (contents not in git); the code is in service/
data/                           train.csv goes here; kept by DVC, git holds only the .dvc pointer files
.dvc/                           DVC settings
figures/                        charts used by WORKBOOK_EXPLAINED.md
research/                       the LLMmap paper (USENIX Security 2025)
requirements.txt                the libraries the notebook needs
Dockerfile, docker-compose.yml, docker/   JupyterLab (B), and Airflow + MLflow + the API (C), in containers
WORKBOOK_EXPLAINED.md           the method and the experiments in more detail
CONTRIBUTING.md                 how to work on it as a team
```

## The earlier pipeline version

An earlier version of this project was a `.py` package with a FastAPI service, a six-task Airflow pipeline, an MLflow registry,
Optuna tuning and 38 tests. It was reduced to the notebook, and then the pieces the course needs came back around the
notebook in a smaller form (this README). The earlier version is still saved under the git tag `archive/pipeline-version`; if the
tag isn't on your copy, the same version is the commit `dfdf20d`, which is in this repository's history (`git checkout dfdf20d`).
It had unit tests, which the current version doesn't.

```bash
git checkout archive/pipeline-version        # look at it (git checkout - to come back)
git push origin archive/pipeline-version     # put the tag on GitHub so it is not only on this computer
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
