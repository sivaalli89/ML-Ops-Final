"""Train the model, check it with prompt-grouped cross-validation, and save it.

    python -m name_that_llm.train --cv            # honest score on prompts the model has never seen
    python -m name_that_llm.train --fit           # train on everything and save artifacts/model.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .features import StyleFeatures
from .group_ops import assign_one_model_each, center_by_group
from .model import EMBEDDING_WEIGHT, ModelBundle, blend

ROOT = Path(__file__).resolve().parents[2]


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


def _run_fold(texts, labels, groups, embeddings, train_rows, test_rows):
    train_embeddings = None if embeddings is None else embeddings[train_rows]
    test_embeddings = None if embeddings is None else embeddings[test_rows]
    bundle = fit_bundle(texts.iloc[train_rows], labels[train_rows], groups[train_rows], train_embeddings)
    style_probs, embedding_probs = bundle.predict_parts(texts.iloc[test_rows], groups[test_rows], test_embeddings)

    candidates = {"style only": style_probs}
    if embedding_probs is not None:
        candidates["style + embeddings"] = blend(style_probs, embedding_probs, bundle.embedding_weight)
    scores = {}
    for name, probs in candidates.items():
        guess = bundle.classes[assign_one_model_each(probs, groups[test_rows])]
        scores[name] = balanced_accuracy_score(labels[test_rows], guess)
    return scores


def cross_validate(texts, labels, groups, embeddings=None, n_jobs=5):
    """Score on whole prompts the model has never seen (5 folds)."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    folds = GroupKFold(5).split(texts, labels, groups)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(texts, labels, groups, embeddings, train_rows, test_rows) for train_rows, test_rows in folds)
    return pd.DataFrame(results)


def load_training_data(data_file, embedding_file=None):
    df = pd.read_csv(data_file).dropna(subset=["text", "model"]).reset_index(drop=True)
    embeddings = np.load(embedding_file) if embedding_file and Path(embedding_file).exists() else None
    return df, embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=ROOT / "data" / "train_with_groups.csv")
    parser.add_argument("--embeddings", default=ROOT / "data" / "embeddings" / "distilbert.npy")
    parser.add_argument("--out", default=ROOT / "artifacts" / "model.joblib")
    parser.add_argument("--cv", action="store_true", help="print grouped cross-validation scores")
    parser.add_argument("--fit", action="store_true", help="train on all rows and save the model")
    args = parser.parse_args()

    df, embeddings = load_training_data(args.data, args.embeddings)
    groups = df["group"].to_numpy()

    if args.cv:
        scores = cross_validate(df["text"], df["model"], groups, embeddings)
        print(pd.DataFrame({"mean": scores.mean(), "std": scores.std()}).round(3))
    if args.fit:
        bundle = fit_bundle(df["text"], df["model"], groups, embeddings)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
