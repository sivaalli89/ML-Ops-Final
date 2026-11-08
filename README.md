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
| Random split instead of by prompt (misleading) | 0.114 |
| Split by prompt, plain model | 0.313 |
| + centering by prompt group | 0.381 |
| + one-model-each assignment | 0.422 |
| + skeleton features | 0.466 |
| Logistic regression tuned a little (C=1) | 0.477 |
| + 10% DistilBERT blend | **0.520** |

These are local estimates on the training data, not scores from the Kaggle competition.

## How to run it

The competition data is not in git. Put `train.csv` in `data/` first (download it from the Kaggle competition page). A full
run takes about 6 to 8 minutes and needs a few GB of memory, because the feature tables are large.

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

**DistilBERT numbers (section 7).** The notebook reads `data/embeddings/distilbert.npy` if it exists. That file is not in git,
so on a fresh copy the notebook computes it once, which needs `sentence-transformers` (it installs PyTorch) and takes a few
minutes on a CPU. With Option A, `pip install sentence-transformers`. With Option B, build with
`INSTALL_EMBEDDINGS=true docker compose up --build` (PowerShell: `$env:INSTALL_EMBEDDINGS="true"; docker compose up --build`),
which makes the image about 1 GB bigger. After the first run the file is reused.

## Where it trains, and what "deployed" means here

Training happens inside the notebook, on whichever Python runs it: your virtual environment (Option A) or the container
(Option B, Python 3.12 with pinned libraries). The notebook measures how well the method works; it does **not** save a
trained model or serve predictions, so nothing is deployed. If you need predictions for the competition, a last cell can train
on all of `train.csv` and write a prediction for each passage in `test.csv`. The archived pipeline version (below) had a
running prediction service.

## What is in the folder

```
notebooks/name_that_llm.ipynb   the whole project, with results saved in it
WORKBOOK_EXPLAINED.md           the method and the experiments in more detail
data/                           train.csv goes here (not in git); embeddings/ is the DistilBERT cache
figures/                        charts used by WORKBOOK_EXPLAINED.md
research/                       the LLMmap paper (USENIX Security 2025)
requirements.txt                the libraries to install
Dockerfile, docker-compose.yml  run the notebook in JupyterLab inside a container (Option B)
CONTRIBUTING.md                 how to work on it as a team
```

## The earlier pipeline version

An earlier version also had a `.py` package, a FastAPI service, an Airflow pipeline, an MLflow model registry with
versioning, Optuna tuning, tests and Docker files for them. It was removed to keep this project to the notebook, but nothing is lost:
it is saved under the git tag `archive/pipeline-version`.

```bash
git checkout archive/pipeline-version        # look at it (git checkout - to come back)
git push origin archive/pipeline-version     # put the tag on GitHub so it is not only on this computer
```

Only the Python side of that version was ever tested. Its Docker images, MLflow server and Airflow pipeline were written but
never built or run.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
