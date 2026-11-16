# What the "Name That LLM" workbook does (under the hood)

This file explains the method and the experiments in more detail. For how to run things, see the [README](README.md).
Everything runs in one notebook, `notebooks/name_that_llm.ipynb`. Your original notebook is in git history (first commit).

## Folder guide

```
ML Ops Final/
├─ README.md, CONTRIBUTING.md   how to run it, how to work on it
├─ WORKBOOK_EXPLAINED.md        this file
├─ notebooks/   name_that_llm.ipynb      the method and the training, with results saved in it
├─ data/        train.csv, test.csv and embeddings/ (cached DistilBERT numbers); kept by DVC, not by git
├─ Dockerfile, docker-compose.yml, docker/   JupyterLab (B), and Airflow + MLflow + the API (C), in containers
├─ dags/        train_pipeline_dag.py: the Airflow DAG (check data, run the notebook, register the model)
├─ pipeline/    register_model.py (MLflow run, registry, promotion) and simulate_drift.py (the drift test)
├─ service/     api.py (the FastAPI service that serves the champion model) and monitoring.py (its drift monitoring)
├─ models/      model.joblib (kept by DVC), model_info.json (version, held-out score, data hash), holdout.csv, optuna_trials.csv
├─ runs/        executed copies of the notebook, one per Airflow run (not in git)
├─ monitoring/  the API's request log and the drift test's results (not in git)
├─ .dvc/        DVC settings; data/*.dvc and models/*.dvc are the pointer files git keeps
├─ figures/     flow chart and charts used in this file
├─ research/    usenixsecurity25-pasquini.pdf
└─ requirements.txt
```

## 0. The whole thing in one picture

![Pipeline flow chart](figures/pipeline_flowchart.png)

Same flow as text (renders as a diagram in GitHub or in VS Code with a Mermaid extension):

```mermaid
flowchart TD
    A["1. DATA<br/>2,699 passages = ~150 prompts x 18 models"] --> B
    B["2. RECOVER PROMPT GROUPS<br/>TF-IDF topic words, LSA, clustering k=150<br/>ARI 0.977"] --> C
    C["3. VALIDATE BY GROUP<br/>hold out whole prompts<br/>random 0.114 vs grouped 0.313"] --> D
    D["4. STYLE FEATURES<br/>char n-grams + word n-grams<br/>+ function-word skeleton (new)"] --> E
    E["5. PROMPT-CENTERING<br/>subtract each group's average<br/>topic cancels, style stays"] --> F
    F["6. CLASSIFIER<br/>logistic regression, 18 probabilities<br/>optional: family first, then model"] --> G
    G["7. ASSIGNMENT<br/>18 passages per prompt get 18 different models"] --> H
    H["8. PREDICTED MODEL"]
```

**Score ladder** (balanced accuracy, grouped CV, chance = 0.056):
plain 0.309 → + centering 0.381 → + assignment 0.422 → + skeleton features 0.466 → logistic regression C=1 0.477 → + 10% DistilBERT blend **0.521**.

**Plain-English summary:** the same 150 questions were put to 18 AIs. The test uses new questions, so we can't learn "topic → AI". Instead we find which passages answer the same question, subtract what they have in common (the topic), and look at what's left: each AI's writing habits. Then we use the rule "each AI answers each question once" to clean up the guesses.

## 1. The problem

**Authorship attribution for LLMs.** Given a short prose passage, predict which of **18 LLMs** wrote it.

| File | Contents |
|---|---|
| `data/train.csv` | 2,699 rows: `id, prompt_id, text, model` |
| `data/test.csv` | only 3 rows here (a tiny sample): `id, text` only (no model, no prompt) |
| `figures/group_recovery.png`, `figures/grouped_cv_baseline.png` | figures from the first version of the notebook |

### The hidden structure
The data was generated as **~150 prompts × 18 models, one answer each** (150 × 18 ≈ 2,700; one model has 149 rows). So passages come in "families": the 18 answers to the same prompt discuss the same topic and differ only in author style.

The **test set uses unseen prompts**. Consequences:

1. A classifier that learns "topic X → model Y" is useless on test. Only style transfers.
2. Honest validation must hold out **whole prompts** (GroupKFold), not random rows.
3. Within one prompt, all 18 answers come from 18 *different* models. That is a constraint we can exploit.

## 2. Pipeline

```
text ──► TF-IDF (content words) ──► LSA 200-d ──► agglomerative clustering (k≈150)
                                                         │
                                          recovered prompt groups  (ARI 0.977)
                                                         │
          ┌──────────────────────────────────────────────┼─────────────────────────┐
          ▼                                              ▼                         ▼
 GroupKFold validation                      prompt-centering of features    Hungarian assignment
 (honest score)                             (remove topic, keep style)      (1 model per slot)
```

### Step A. Recover the prompt groups (notebook section 2)
- **Topic vector:** TF-IDF restricted to *content* words (stopwords removed, `max_df=0.05` drops anything common, alphabetic ≥3 letters), then **TruncatedSVD to 200 dims** (LSA) and L2-normalise. Style words are deliberately removed so passages cluster by *subject*, not author.
- **Clustering:** `AgglomerativeClustering(metric="cosine", linkage="average")`.
- **Choosing k without labels:** `quality()` uses the known structure: a perfect cluster has ~18 passages and no model twice. Sweep result:

| k | clusters w/ a repeated model | clusters of size 15–21 | singletons |
|---|---|---|---|
| 123 | 17.9% | 82% | 0 |
| 146 | 4.8% | 95.9% | 1 |
| **150** | **2.7%** | **95.3%** | 2 |
| 168 | 2.4% | 85% | 13 |
| 225 | 0.9% | 58% | 47 |

  Too few clusters → different prompts merge (repeats). Too many → prompts split (singletons). k = 146–150 balances both.
- **Grading:** train.csv happens to include the true `prompt_id`, so **ARI = 0.977** against it (1.0 = perfect). The label-free purity metrics agree, which is what makes the method trustworthy for the test set, where no prompt ids exist.

### Step B. Why validation must be grouped
Baseline model: char 2–5-gram TF-IDF + class-balanced logistic regression, scored by balanced accuracy (chance = 1/18 = 0.056).

| Split | Balanced accuracy |
|---|---|
| Random 5-fold | **0.114** |
| GroupKFold (recovered prompts) | **0.313** |

Random CV scores *lower*, not higher, which is the opposite of normal leakage. Reason: with random splits, the other 17 answers to the held-out passage's prompt sit in training labelled with *other* models. The classifier learns "this topic ⇒ not this model" and is pushed away from the truth. Either way a random split misestimates test performance, so all later numbers use GroupKFold.

### Step C. Extra style features, a negative result
Adds function-word rates, punctuation/markdown counts, sentence/word length, and the first three words. Score **0.283 < 0.313**. More hand-made features hurt with only ~2.7k samples; the markdown features are almost always zero here (plain prose), while em-dash/semicolon/colon appear in 39–52% of passages.

The "top phrases" table (one logistic regression on word 1–2-grams) shows each model's most distinctive tokens, e.g. Gemini → "profound", "fundamental"; GPT-5.6 variants → "may", "merely". Treat it as tendencies, since many top tokens are function words.

### Step D. The main improvement (centering and assignment)
Features: char n-gram TF-IDF (20k) + word 1–2-gram TF-IDF (15k), densified, fit on the training fold only.

1. **Prompt-centering.** For every prompt group, subtract that group's mean feature vector from each member. All 18 answers share the topic, so the topic's contribution is the group mean and cancels; what remains is each author's deviation, which is style.
2. **One-to-one assignment.** Take the classifier's probabilities `P` for the passages in a group, build cost `-log P`, and solve the assignment problem (`scipy.optimize.linear_sum_assignment`, a.k.a. Hungarian algorithm). That forces the 18 passages to get 18 distinct models, maximising total likelihood instead of taking each argmax independently.

| Variant (5-fold GroupKFold) | Mean | Std |
|---|---|---|
| plain | 0.309 | 0.007 |
| + centering | 0.381 | 0.013 |
| + centering + assignment | **0.422** | 0.010 |

So each trick adds roughly +0.07 and +0.04 respectively; ~7.5× chance.

### Step E. Two-stage classifier and "skeleton" features (an experiment, not in the notebook)
Same 5-fold GroupKFold, so numbers compare directly with Step D.

- **New feature block, "function-word skeleton".** Every word outside the 150 most common becomes `W`; punctuation stays. `The cat sat; it was warm.` → `the W W ; it was W .` Its 1–4-grams describe sentence rhythm and syntax with almost no topic.
- **Two-stage classifier.** Stage 1 guesses the family (Claude, Gemini, GPT, GLM, Kimi, Qwen, from the model-name prefix). Stage 2 uses a separate classifier per family. Final probability = P(family) × P(model | family), then assignment.

| Features | Classifier | Centred | Centred + assignment |
|---|---|---|---|
| Old (char + word) | flat | 0.381 | 0.422 |
| Old | two-stage | 0.376 | 0.433 |
| **New (+ skeleton)** | flat | **0.431** | **0.466** |
| New | two-stage | 0.428 | 0.465 |

- **Skeleton features help:** about +0.045, better in all 5 folds (+0.02 to +0.064).
- **Two-stage doesn't clearly help:** +0.011 with old features, 0 with new, and it swings from −0.033 to +0.028 between folds. Stage 1 names the right company only ~73% of the time, and those mistakes can't be repaired in stage 2.
- **Recommended setup so far:** new features + flat classifier + centering + assignment = 0.466.

### Step F. 2D charts of the models (what they do and don't show)
| Chart | File | What it is | Explainable? |
|---|---|---|---|
| PCA, topic | `figures/models_2d_topic.png` | Models fully mixed: topic says nothing about author | Axes = subject matter (PC1 1.5%, PC2 1.3% of variance) |
| PCA, style | `figures/models_2d_style.png` | Gemini separates; the rest overlap | Axes readable via n-gram loadings (PC1 0.8%, PC2 0.7%) |
| t-SNE, style | `figures/models_2d_tsne.png` | Gemini separates, mild order among the others | **No**: axes and distances between clusters have no meaning |
| LDA, style | `figures/models_2d_lda.png` | Three chunks: Gemini / GPT+Kimi+Qwen / Claude+GLM | Yes (linear), but fitted in-sample so it looks tidier than reality |

Takeaway: the models cluster into families, while telling models apart inside a family is the hard part. The 2D charts show only a thin slice (PCA axes cover ~1.5% of style variance), so overlap in 2D doesn't mean the models are inseparable in the full feature space. For honest accuracy, use the cross-validated scores, not how clean a picture looks.

### Step G. Better models and ensembles (experiments, with the winner in the notebook)
Same 5-fold GroupKFold. Baseline = skeleton features, centring, assignment.

| Idea | Score | Verdict |
|---|---|---|
| Logistic regression C=1 instead of 5 | 0.477 | small gain, kept |
| Linear SVM / averaging LR + SVM | 0.466 | no help |
| Averaging per-feature-block models | 0.474 | no help (same kind of signal) |
| MiniLM embeddings alone | 0.205 | weak |
| DistilBERT embeddings alone | 0.396 | weaker than TF-IDF |
| TF-IDF + 50/50 embeddings | worse | hurts |
| **TF-IDF + 10% DistilBERT** | **0.521** | better in 5/5 folds, kept |
| TF-IDF + 20% / 30% DistilBERT | 0.503 / 0.473 | too much weight hurts |

The DistilBERT weight was picked on the same folds, so 0.521 is a little optimistic (a realistic gain is perhaps +0.02 to +0.04). This blend is section 7 of the notebook. These scores are local cross-validation estimates, not Kaggle scores.

Tuning with Optuna did **not** help once checked honestly, and that has now been seen twice.
**First check (earlier pipeline version, see section 5).** Scored on the folds that picked the settings it reached 0.532, but
tuning inside each fold and testing on untouched prompts gave 0.520 for both the hand-picked and the tuned settings: in four
of five folds the search simply kept the hand-picked values (6 trials and 3 inner folds).
**Second check (the notebook now, sections 8 and 9).** 20% of the prompts are held out first. Optuna searches `style_c`,
`embedding_c` and `embedding_weight` on the other 80%, starting from the hand-picked values. None of its 20 trials scored better
on the cross-validation folds (0.517 for the hand-picked values, and the best of 20 was the same point), so the deployed
model uses the hand-picked settings, and its score on the held-out prompts is 0.478. That number comes from about 30 prompts and a
model trained on 80% of the data, so it should be read as "about 0.48", not as a drop from 0.521.

## 3. Things to know
- **Test-time requirement.** Centering and assignment both need the test passages grouped by prompt. You must cluster `data/test.csv` with the same recipe (it's small, so check the cluster sizes carefully: assignment assumes ≤18 per group, and 18 distinct models per group).
- **`prompt_id` is only used for grading** the clustering (ARI). The pipeline itself never reads it.
- **The research PDF (`research/usenixsecurity25-pasquini.pdf`) is *LLMmap* (USENIX Security 2025).** It fingerprints 42 LLM versions at >95% accuracy, but with chosen probe questions and ~8 responses per model, so it is an easier setup than one free-form passage. Ideas that transfer: e5 embeddings (`multilingual-e5-large-instruct`) feeding a small transformer, contrastive training, and pooling several samples (your assignment step does a version of this). It does **not** contain the markdown, function-word or first-word features, and its Fig. 6 is a 2D map of learned signatures, not a phrase list. So the notebook's comments "style features from the paper" and "paper Fig. 6" must refer to some other paper, which I haven't identified.
- **Not done yet:** contrastive training, a run on the real test set, and e5 embeddings in the blend. (The skeleton features and the DistilBERT blend are in the notebook; the two-stage classifier was dropped.)
- **Next steps listed by the author:** e5 + contrastive training, two-stage (organization → model).

## 4. How to re-run
See the [README](README.md): run the notebook with Python on your computer or in a Docker container, or run the whole system (Airflow, MLflow, the API) with one compose command. The notebook reads `../data/train.csv` and builds dense
~55k-column matrices, so give it a few GB of RAM. A full run takes about 26 minutes under Airflow (measured; the Optuna section and the extra fits roughly doubled the time of the earlier 12 to 15 minutes in a container, and 6 to 7 on a local Python); the README explains why.

## 5. The earlier pipeline version (archived)

Before the project was reduced to the notebook, it also had:

| Part | What it did |
|---|---|
| A `.py` package (`src/name_that_llm/`) | the notebook's steps as reusable code, run from the command line |
| Optuna tuning | searched three settings, scored on held-out prompts |
| An MLflow model registry | kept every trained model as a numbered version with a semantic version tag, and promoted a new one to "champion" only if its score was not worse |
| A larger FastAPI service | `/predict` took a batch of passages and returned a guessed model for each (a smaller one, `service/api.py`, is back; see section 6) |
| An Airflow pipeline | group, tune, train, evaluate and register as six tasks in order |
| Tests and Docker files | 38 tests on made-up data; images for the API, MLflow and Airflow |

All of it is saved under the git tag `archive/pipeline-version`. To look at it: `git checkout archive/pipeline-version`
(`git checkout -` to come back). The tag exists only on the computer that made it until someone runs
`git push origin archive/pipeline-version`; without it, the same version is the commit `dfdf20d`, which is in this repository's history.

What was and was not verified: the Python code and the tests were run on the real data, and the registry was exercised
against a local MLflow store. The Docker images, the MLflow server and the Airflow pipeline were written but never built or run.
The one lasting finding is the honest tuning result in Step G: Optuna did not improve on the hand-picked settings.

## 6. Around the notebook: trigger, tracking, registry, serving, monitoring, versioning

The notebook does the thinking and the training; everything below runs it, records it or serves what it saved. None of it
changes the method.

**Trigger (Airflow).** A notebook does nothing until someone runs it. The DAG `train_pipeline` (`dags/train_pipeline_dag.py`)
runs it on a button press or a schedule, logs each run and keeps an executed copy of the notebook in `runs/`. Three tasks:
check the data is there, run the notebook, register the model. The DAG has no model code.

**Held-out prompts and Optuna (notebook sections 8 and 9).** 20% of the prompts are set aside before anything is tuned. Optuna
searches three settings (`style_c`, `embedding_c`, `embedding_weight`) on the other 80%, using 5 folds split by prompt. Fitting
the slow style model is done once per candidate C and fold, and Optuna chooses among those saved results, which is why the
search is cheap. The final model is trained on the 80% and scored on the held-out 20%, which no choice depended on. The model
is deliberately not refitted on everything afterwards: the held-out prompts have to stay unseen so they can validate the
deployed service and serve as the monitoring baseline.

**Experiment tracking and registry (MLflow).** The last Airflow task, `pipeline/register_model.py`, logs a run (settings and
scores, one nested run per Optuna trial, the model files and the executed notebook), registers the model as a new version and
tags it with a semantic version: the notebook's `MODEL_VERSION` if it is free, else the next patch number. The alias
`champion` moves to the new version only if its held-out score is at least as good as the champion's. A worse model is kept
in the registry but not served. This is the part the first version had and that was removed with the `.py` package; it is back
in a smaller form that reads the notebook's output files instead of importing the notebook's code.

**Serving (FastAPI).** `service/api.py` loads the champion from the registry and exposes `POST /predict`, `GET /model`, `GET /monitor`,
`GET /dashboard` and `GET /health`. It repeats the notebook's helper functions (centering, assignment, blending) for one
prompt's passages, since the saved model holds only scikit-learn objects. It asks the registry for the champion every 10
seconds at most, so a newly promoted model is live without a restart.

**Monitoring (`service/monitoring.py`).** Production has no labels, so the service watches its inputs and the model's confidence:
eight numbers per passage (lengths, word length, punctuation, unknown words, known word pairs, similarity to the other passages
in the request, confidence). The baseline is the same numbers for half of the held-out prompts, computed by the service itself
when it loads a model. Evidently 0.7.23's data drift report compares recent passages with it, using the normalised Wasserstein
distance (how far the values moved, in units of the baseline's spread; drift above 1.0). A p-value test was tried first and
rejected: the passages of one prompt resemble each other, so they are not independent rows, and it raised false alarms on clean
data (6 of 10 samples). Malformed requests are counted too. The other half of the held-out prompts is the clean data for
`pipeline/simulate_drift.py`, which also corrupts it in five ways (truncating, random characters, shuffled words, mixed prompts,
broken request schema) and records what the monitor reports. Results are in the README.

**Data versioning.** `train.csv` is tracked by DVC, which stores the file by its content hash and gives git a small pointer.
Changing the data produces a new hash and, after `dvc add` and a commit, a new version (`data-v2`). Going back is a git
checkout plus `dvc checkout`.

**Model file versioning.** Notebook section 9 saves `models/model.joblib`, `model_info.json`, `holdout.csv` and
`optuna_trials.csv`. The info file holds the version number, the held-out and cross-validated scores, the settings and the hash
of the training data, so a model always says which data produced it. The model file is versioned like the data (`dvc add`,
commit, tag `model-v0.2.0`). DVC and tags answer "what existed when"; the MLflow registry answers "what is serving now".

**What this does not do.** It has no unit tests (the archived version had 38), no alerting beyond the dashboard and the JSON
status, no login on any page, and it runs on one machine (SQLite for Airflow and MLflow, no shared storage). See *What has been
tested* in the README for what was and was not run.
