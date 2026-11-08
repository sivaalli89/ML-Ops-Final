"""Train the model, check it with prompt-grouped cross-validation, and save it.

    python -m name_that_llm.train --cv            # honest score on prompts the model has never seen
    python -m name_that_llm.train --fit           # train on everything and save artifacts/model.joblib
    python -m name_that_llm.train --cv --fit --params artifacts/best_params.json    # use tuned settings
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import ROOT
from .features import StyleFeatures
from .grouping import assign_one_model_each, center_by_group
from .model import EMBEDDING_WEIGHT, ModelBundle, blend

# the settings that worked best by hand; tune.py searches around these
DEFAULT_PARAMS = {"style_c": 1.0, "embedding_c": 0.1, "embedding_weight": EMBEDDING_WEIGHT}


def fit_bundle(texts, labels, groups, embeddings=None, style_c=1, embedding_c=0.1, embedding_weight=EMBEDDING_WEIGHT):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    groups = np.asarray(groups)

    style = StyleFeatures().fit(texts)
    style_x = center_by_group(style.transform(texts), groups)
    style_model = LogisticRegression(C=style_c, class_weight="balanced", max_iter=600).fit(style_x, labels)
    bundle = ModelBundle(classes, style, style_model, embedding_weight=embedding_weight)

    if embeddings is not None:
        embedding_x = center_by_group(embeddings, groups)
        bundle.embedding_scaler = StandardScaler().fit(embedding_x)
        bundle.embedding_model = LogisticRegression(C=embedding_c, class_weight="balanced", max_iter=2000).fit(
            bundle.embedding_scaler.transform(embedding_x), labels)
    return bundle


def score_bundle(bundle, texts, labels, groups, embeddings=None):
    """Balanced accuracy of a trained bundle on passages it has never seen, with centering and assignment."""
    style_probs, embedding_probs = bundle.predict_parts(texts, groups, embeddings)
    candidates = {"style only": style_probs}
    if embedding_probs is not None:
        candidates["style + embeddings"] = blend(style_probs, embedding_probs, bundle.embedding_weight)
    scores = {}
    for name, probs in candidates.items():
        guess = bundle.classes[assign_one_model_each(probs, groups)]
        scores[name] = balanced_accuracy_score(labels, guess)
    return scores


def _run_fold(texts, labels, groups, embeddings, train_rows, test_rows, params):
    train_embeddings = None if embeddings is None else embeddings[train_rows]
    test_embeddings = None if embeddings is None else embeddings[test_rows]
    bundle = fit_bundle(texts.iloc[train_rows], labels[train_rows], groups[train_rows], train_embeddings, **params)
    return score_bundle(bundle, texts.iloc[test_rows], labels[test_rows], groups[test_rows], test_embeddings)


def cross_validate(texts, labels, groups, embeddings=None, n_jobs=5, params=None):
    """Score on whole prompts the model has never seen (5 folds)."""
    params = params or {}
    labels, groups = np.asarray(labels), np.asarray(groups)
    folds = GroupKFold(5).split(texts, labels, groups)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(texts, labels, groups, embeddings, train_rows, test_rows, params)
        for train_rows, test_rows in folds)
    return pd.DataFrame(results)


def load_training_data(data_file, embedding_file=None):
    df = pd.read_csv(data_file).dropna(subset=["text", "model"]).reset_index(drop=True)
    embeddings = np.load(embedding_file) if embedding_file and Path(embedding_file).exists() else None
    return df, embeddings


def summarize_scores(scores):
    """Mean and spread of each variant over the folds (scores is what cross_validate returns)."""
    return pd.DataFrame({"mean": scores.mean(), "std": scores.std()}).round(3)


def save_metrics(summary, params, path):
    """Write the headline score and the settings used to a json file (the registry step reads it)."""
    headline = "style + embeddings" if "style + embeddings" in summary.index else "style only"
    metrics = {"balanced_accuracy": float(summary.loc[headline, "mean"]), "variant": headline,
               "scores": summary.to_dict("index"), "settings": params or DEFAULT_PARAMS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2))
    return metrics


def load_params(path):
    """Read settings written by tune.py (or any json with style_c / embedding_c / embedding_weight)."""
    saved = json.loads(Path(path).read_text())
    saved = saved.get("best_params", saved)
    return {key: saved[key] for key in DEFAULT_PARAMS if key in saved}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=ROOT / "data" / "train_with_groups.csv")
    parser.add_argument("--embeddings", default=ROOT / "data" / "embeddings" / "distilbert.npy")
    parser.add_argument("--out", default=ROOT / "artifacts" / "model.joblib")
    parser.add_argument("--params", help="json file with tuned settings (from name_that_llm.tune)")
    parser.add_argument("--metrics-out", help="with --cv: also write the scores to this json file")
    parser.add_argument("--cv", action="store_true", help="print grouped cross-validation scores")
    parser.add_argument("--fit", action="store_true", help="train on all rows and save the model")
    args = parser.parse_args()

    df, embeddings = load_training_data(args.data, args.embeddings)
    groups = df["group"].to_numpy()
    params = load_params(args.params) if args.params else {}
    if params:
        print("using settings:", params)

    if args.cv:
        scores = cross_validate(df["text"], df["model"], groups, embeddings, params=params)
        summary = summarize_scores(scores)
        print(summary)
        if args.metrics_out:
            save_metrics(summary, params, args.metrics_out)
            print("saved", args.metrics_out)
    if args.fit:
        bundle = fit_bundle(df["text"], df["model"], groups, embeddings, **params)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
