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

```bash
git clone https://github.com/sivaalli89/ML-Ops-Final.git
cd ML-Ops-Final
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The competition data is not in git. Put `train.csv` in `data/` (download it from the Kaggle competition page). Then open the
notebook (`jupyter lab notebooks/name_that_llm.ipynb`, or open it in VS Code) and run it top to bottom. It takes about
6 to 8 minutes and needs a few GB of memory, because the feature tables are large.

**DistilBERT numbers (section 7).** The notebook reads `data/embeddings/distilbert.npy` if it exists. That file is not in git
either, so on a fresh copy the notebook computes it once, which needs `pip install sentence-transformers` (it installs PyTorch)
and takes a few minutes on a CPU. After that the file is reused.

## What is in the folder

```
notebooks/name_that_llm.ipynb   the whole project, with results saved in it
WORKBOOK_EXPLAINED.md           the method and the experiments in more detail
data/                           train.csv goes here (not in git); embeddings/ is the DistilBERT cache
figures/                        charts used by WORKBOOK_EXPLAINED.md
research/                       the LLMmap paper (USENIX Security 2025)
requirements.txt                the libraries to install
CONTRIBUTING.md                 how to work on it as a team
```

## The earlier pipeline version

An earlier version also had a `.py` package, a FastAPI service, an Airflow pipeline, an MLflow model registry with
versioning, Optuna tuning, tests and Docker files. It was removed to keep this project to the notebook, but nothing is lost:
it is saved under the git tag `archive/pipeline-version`.

```bash
git checkout archive/pipeline-version        # look at it (git checkout - to come back)
git push origin archive/pipeline-version     # put the tag on GitHub so it is not only on this computer
```

Only the Python side of that version was ever tested. Its Docker images, MLflow server and Airflow pipeline were written but
never built or run.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
