"""Tune the model's settings with Optuna, always scoring on whole prompts the model has not seen.

    python -m name_that_llm.tune --trials 15            # tune on all the data, save artifacts/best_params.json
    python -m name_that_llm.tune --nested --trials 8    # honest check: tune inside each outer fold, then test

Three settings are searched: the strength of the style classifier (style_c), the strength of the embedding
classifier (embedding_c), and how much the embedding classifier counts in the blend (embedding_weight).
The expensive part (building the text features) is done once per fold; every trial only refits the classifiers.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .features import StyleFeatures
from .group_ops import assign_one_model_each, center_by_group
from .model import blend
from .paths import ROOT
from .train import DEFAULT_PARAMS, fit_bundle, load_training_data, score_bundle


@dataclass
class FoldData:
    """One cross-validation fold, with the features already built."""
    style_train: np.ndarray
    style_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups_test: np.ndarray
    classes: np.ndarray
    embed_train: np.ndarray = None
    embed_test: np.ndarray = None


def _prepare_fold(texts, labels, groups, embeddings, train_rows, test_rows):
    style = StyleFeatures().fit(texts.iloc[train_rows])
    fold = FoldData(
        style_train=center_by_group(style.transform(texts.iloc[train_rows]), groups[train_rows]),
        style_test=center_by_group(style.transform(texts.iloc[test_rows]), groups[test_rows]),
        y_train=labels[train_rows],
        y_test=labels[test_rows],
        groups_test=groups[test_rows],
        classes=np.unique(labels[train_rows]),
    )
    if embeddings is not None:
        train_e = center_by_group(embeddings[train_rows], groups[train_rows])
        test_e = center_by_group(embeddings[test_rows], groups[test_rows])
        scaler = StandardScaler().fit(train_e)
        fold.embed_train, fold.embed_test = scaler.transform(train_e), scaler.transform(test_e)
    return fold


def prepare_folds(texts, labels, groups, embeddings=None, n_splits=5, n_jobs=1):
    splits = GroupKFold(n_splits).split(texts, labels, groups)
    return Parallel(n_jobs=n_jobs)(
        delayed(_prepare_fold)(texts, labels, groups, embeddings, train_rows, test_rows)
        for train_rows, test_rows in splits)


def score_fold(fold, style_c, embedding_c=None, embedding_weight=0.0):
    """Fit both classifiers on the fold's training part and score the held-out prompts."""
    style_probs = LogisticRegression(C=style_c, class_weight="balanced", max_iter=600).fit(
        fold.style_train, fold.y_train).predict_proba(fold.style_test)
    probs = style_probs
    if fold.embed_train is not None and embedding_weight > 0:
        embedding_probs = LogisticRegression(C=embedding_c, class_weight="balanced", max_iter=2000).fit(
            fold.embed_train, fold.y_train).predict_proba(fold.embed_test)
        probs = blend(style_probs, embedding_probs, embedding_weight)
    guess = fold.classes[assign_one_model_each(probs, fold.groups_test)]
    return balanced_accuracy_score(fold.y_test, guess)


def tune(texts, labels, groups, embeddings=None, n_trials=15, n_splits=5, seed=0, n_jobs=5, verbose=True):
    """Search the settings. Returns the best settings and their cross-validated score."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)

    labels, groups = np.asarray(labels), np.asarray(groups)
    folds = prepare_folds(texts, labels, groups, embeddings, n_splits, n_jobs)
    with_embeddings = embeddings is not None

    def objective(trial):
        params = {"style_c": trial.suggest_float("style_c", 0.3, 10, log=True)}
        if with_embeddings:
            params["embedding_c"] = trial.suggest_float("embedding_c", 0.01, 1, log=True)
            params["embedding_weight"] = trial.suggest_float("embedding_weight", 0.0, 0.3)
        scores = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(score_fold)(fold, **params) for fold in folds)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    # trial 0 = the hand-picked settings, so we can see whether tuning really beats them
    study.enqueue_trial({key: value for key, value in DEFAULT_PARAMS.items() if with_embeddings or key == "style_c"})
    study.optimize(objective, n_trials=n_trials)
    return {
        "best_params": study.best_params,
        "cv_score_tuned": study.best_value,
        "cv_score_default": study.trials[0].value,
        "n_trials": n_trials,
    }


def nested_cv(texts, labels, groups, embeddings=None, n_trials=8, n_outer=5, n_inner=3, n_jobs=3, seed=0):
    """Honest comparison of tuned versus default settings.

    The search runs only on each outer fold's training prompts; the outer test prompts are used once, at the end.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    rows = []
    for number, (train_rows, test_rows) in enumerate(GroupKFold(n_outer).split(texts, labels, groups), start=1):
        train_embeddings = None if embeddings is None else embeddings[train_rows]
        test_embeddings = None if embeddings is None else embeddings[test_rows]
        found = tune(texts.iloc[train_rows], labels[train_rows], groups[train_rows], train_embeddings,
                     n_trials, n_inner, seed, n_jobs, verbose=False)
        scores = {}
        for name, params in [("default", {}), ("tuned", found["best_params"])]:
            bundle = fit_bundle(texts.iloc[train_rows], labels[train_rows], groups[train_rows], train_embeddings, **params)
            result = score_bundle(bundle, texts.iloc[test_rows], labels[test_rows], groups[test_rows], test_embeddings)
            scores[name] = result.get("style + embeddings", result["style only"])
        rows.append(scores)
        print(f"outer fold {number}: default {scores['default']:.3f}  tuned {scores['tuned']:.3f}  {found['best_params']}", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=ROOT / "data" / "train_with_groups.csv")
    parser.add_argument("--embeddings", default=ROOT / "data" / "embeddings" / "distilbert.npy")
    parser.add_argument("--out", default=ROOT / "artifacts" / "best_params.json")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=5, help="folds scored in parallel")
    parser.add_argument("--nested", action="store_true", help="honest check instead of a normal search")
    args = parser.parse_args()

    df, embeddings = load_training_data(args.data, args.embeddings)
    groups = df["group"].to_numpy()

    if args.nested:
        scores = nested_cv(df["text"], df["model"], groups, embeddings, n_trials=args.trials, seed=args.seed,
                           n_jobs=min(args.jobs, 3))
        print(pd.DataFrame({"mean": scores.mean(), "std": scores.std()}).round(3))
        return

    result = tune(df["text"], df["model"], groups, embeddings, args.trials, seed=args.seed, n_jobs=args.jobs)
    result["note"] = ("scores come from the same folds that picked the settings, so they are a little optimistic; "
                      "run with --nested for an honest comparison")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
