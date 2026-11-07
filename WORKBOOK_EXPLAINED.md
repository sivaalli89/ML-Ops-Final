# What the "Name That LLM" workbook does (under the hood)

Files: `notebooks/name_that_llm_groups (1).ipynb` (your original, unchanged) and `notebooks/name_that_llm_simple.ipynb` (short rewrite with the improvements).

## Folder guide

```
ML Ops Final/
├─ WORKBOOK_EXPLAINED.md   this file
├─ data/        train.csv, test.csv, train_with_groups.csv (made by the notebooks)
├─ notebooks/   name_that_llm_groups (1).ipynb      your original (unchanged)
│               name_that_llm_simple.ipynb          short rewrite, includes the skeleton features
├─ scripts/     two_stage_pipeline.py, plot_models_2d.py, plot_models_2d_nonpca.py
├─ figures/     flow chart, 2D model charts, notebook plots
├─ results/     cluster_sweep.csv, two_stage_results.csv
└─ research/    usenixsecurity25-pasquini.pdf
```

Start with `notebooks/name_that_llm_simple.ipynb`. Scripts can be run from any folder.
Your original notebook still reads `train.csv` from its own folder: to run it from `notebooks/`, change its `TRAIN_PATH` to `"../data/train.csv"`.

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
plain 0.309 → + centering 0.381 → + assignment 0.422 → + skeleton features **0.466**.

**Plain-English summary:** the same 150 questions were put to 18 AIs. The test uses new questions, so we can't learn "topic → AI". Instead we find which passages answer the same question, subtract what they have in common (the topic), and look at what's left: each AI's writing habits. Then we use the rule "each AI answers each question once" to clean up the guesses.

## 1. The problem

**Authorship attribution for LLMs.** Given a short prose passage, predict which of **18 LLMs** wrote it.

| File | Contents |
|---|---|
| `data/train.csv` | 2,699 rows: `id, prompt_id, text, model` |
| `data/test.csv` | only 3 rows here (a tiny sample): `id, text` only (no model, no prompt) |
| `data/train_with_groups.csv` | train + a recovered `group` column (output of this notebook) |
| `results/cluster_sweep.csv` | quality table for each number of clusters tried |
| `figures/group_recovery.png`, `figures/grouped_cv_baseline.png` | the two figures the notebook saves |

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

### Step A. Recover the prompt groups (sections 2–4)
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

### Step B. Why validation must be grouped (section 5)
Baseline model: char 2–5-gram TF-IDF + class-balanced logistic regression, scored by balanced accuracy (chance = 1/18 = 0.056).

| Split | Balanced accuracy |
|---|---|
| Random 5-fold | **0.114** |
| GroupKFold (recovered prompts) | **0.313** |

Random CV scores *lower*, not higher, which is the opposite of normal leakage. Reason: with random splits, the other 17 answers to the held-out passage's prompt sit in training labelled with *other* models. The classifier learns "this topic ⇒ not this model" and is pushed away from the truth. Either way a random split misestimates test performance, so all later numbers use GroupKFold.

### Step C. Extra style features, a negative result
Adds function-word rates, punctuation/markdown counts, sentence/word length, and the first three words. Score **0.283 < 0.313**. More hand-made features hurt with only ~2.7k samples; the markdown features are almost always zero here (plain prose), while em-dash/semicolon/colon appear in 39–52% of passages.

The "top phrases" table (one logistic regression on word 1–2-grams) shows each model's most distinctive tokens, e.g. Gemini → "profound", "fundamental"; GPT-5.6 variants → "may", "merely". Treat it as tendencies, since many top tokens are function words.

### Step D. The main improvement (final code cell)
Features: char n-gram TF-IDF (20k) + word 1–2-gram TF-IDF (15k), densified, fit on the training fold only.

1. **Prompt-centering.** For every prompt group, subtract that group's mean feature vector from each member. All 18 answers share the topic, so the topic's contribution is the group mean and cancels; what remains is each author's deviation, which is style.
2. **One-to-one assignment.** Take the classifier's probabilities `P` for the passages in a group, build cost `-log P`, and solve the assignment problem (`scipy.optimize.linear_sum_assignment`, a.k.a. Hungarian algorithm). That forces the 18 passages to get 18 distinct models, maximising total likelihood instead of taking each argmax independently.

| Variant (5-fold GroupKFold) | Mean | Std |
|---|---|---|
| plain | 0.309 | 0.007 |
| + centering | 0.381 | 0.013 |
| + centering + assignment | **0.422** | 0.010 |

So each trick adds roughly +0.07 and +0.04 respectively; ~7.5× chance.

### Step E. Two-stage classifier and "skeleton" features (added later, `scripts/two_stage_pipeline.py`)
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

## 3. Things to know
- **Test-time requirement.** Centering and assignment both need the test passages grouped by prompt. You must cluster `data/test.csv` with the same recipe (it's small, so check the cluster sizes carefully: assignment assumes ≤18 per group, and 18 distinct models per group).
- **`prompt_id` is only used for grading** the clustering (ARI). The pipeline itself never reads it.
- **Optional e5 embeddings.** `E5_PATH` swaps TF-IDF+LSA for a sentence-transformer; not run here. It may help with the ~5% of clusters that still merge.
- **The research PDF (`research/usenixsecurity25-pasquini.pdf`) is *LLMmap* (USENIX Security 2025).** It fingerprints 42 LLM versions at >95% accuracy, but with chosen probe questions and ~8 responses per model, so it is an easier setup than one free-form passage. Ideas that transfer: e5 embeddings (`multilingual-e5-large-instruct`) feeding a small transformer, contrastive training, and pooling several samples (your assignment step does a version of this). It does **not** contain the markdown, function-word or first-word features, and its Fig. 6 is a 2D map of learned signatures, not a phrase list. So the notebook's comments "style features from the paper" and "paper Fig. 6" must refer to some other paper, which I haven't identified.
- **Not done yet:** e5 embeddings (needs torch and a ~2 GB model download), contrastive training, and merging the skeleton features into the notebook itself. They exist only in `scripts/two_stage_pipeline.py`.
- **Next steps listed by the author:** e5 + contrastive training, two-stage (organization → model), MLflow logging of `cv_scheme`, `k`, `cluster_quality`.

## 4. How to re-run
Needs `pandas, numpy, scikit-learn, scipy, matplotlib`. Put `data/train.csv` next to the notebook and run top to bottom (the GroupKFold cells use `n_jobs=-1` and the last cell builds dense ~35k-column matrices, so give it a few GB of RAM).
