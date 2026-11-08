"""Find which passages answer the same prompt, and use those groups (centering, one-model-each assignment)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

N_MODELS = 18   # every model answers every prompt once, so a full group has 18 passages
MIN_BATCH = 54  # below this (3 prompts) the topic words are too few to cluster reliably


def cluster_passages(texts, n_groups=None, n_models=N_MODELS, seed=42):
    """Return one group number per passage.

    Only rare content words are used, so passages cluster by subject and not by author.
    """
    texts = list(texts)
    if n_groups is None:
        n_groups = max(1, round(len(texts) / n_models))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        sublinear_tf=True,
        min_df=2,
        max_df=0.05,
    )
    word_weights = vectorizer.fit_transform(texts)

    # squash to at most 200 numbers per passage (this step is called LSA)
    n_parts = min(200, word_weights.shape[1] - 1, word_weights.shape[0] - 1)
    topic_vectors = normalize(TruncatedSVD(n_parts, random_state=seed).fit_transform(word_weights))

    clusterer = AgglomerativeClustering(n_clusters=n_groups, metric="cosine", linkage="average")
    return clusterer.fit_predict(topic_vectors)


def cluster_quality(groups, models=None, prompt_ids=None):
    """How good are the groups? A good group has about 18 passages and never the same model twice."""
    groups = np.asarray(groups)
    sizes = np.bincount(groups)
    quality = {"n_groups": len(sizes), "median_size": float(np.median(sizes))}
    if models is not None:
        models = np.asarray(models)
        repeats = [len(set(models[groups == g])) < (groups == g).sum() for g in np.unique(groups)]
        quality["share_with_repeat"] = float(np.mean(repeats))
    if prompt_ids is not None:
        quality["ari"] = float(adjusted_rand_score(prompt_ids, groups))
    return quality


def center_by_group(features, groups):
    """Subtract each group's average row. All passages in a group share the topic, so it cancels out."""
    centered = features.copy()
    for g in np.unique(groups):
        rows = (groups == g)
        centered[rows] = features[rows] - features[rows].mean(axis=0)
    return centered


def assign_one_model_each(probs, groups):
    """Inside a group each model should appear once, so find the best one-to-one matching."""
    n_models = probs.shape[1]
    best_guess = probs.argmax(axis=1)
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        cost = -np.log(probs[rows] + 1e-9)          # low cost = likely
        if len(rows) > n_models:                     # a merged cluster: let models repeat
            cost = np.tile(cost, int(np.ceil(len(rows) / n_models)))
        row_ids, model_ids = linear_sum_assignment(cost)
        best_guess[rows[row_ids]] = model_ids % n_models
    return best_guess


def main():
    """python -m name_that_llm.grouping : make data/train_with_groups.csv from data/train.csv"""
    import argparse
    import pandas as pd

    from . import ROOT

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data", default=ROOT / "data" / "train.csv")
    parser.add_argument("--out", default=ROOT / "data" / "train_with_groups.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "model"]).reset_index(drop=True)
    df["group"] = cluster_passages(df["text"], n_models=df["model"].nunique())
    prompt_ids = df["prompt_id"] if "prompt_id" in df else None   # only used to grade the groups
    print(cluster_quality(df["group"], df["model"], prompt_ids))
    df.to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
