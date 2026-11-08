"""Model versioning with the MLflow Model Registry.

    python -m name_that_llm.registry        # log the latest training run, register the model, promote if better

Every trained model becomes a numbered registry version with a semantic version tag (for example 0.1.3).
The registry's "champion" alias points at the version that serves traffic. A new version only becomes the
champion if its cross-validated score is at least as good as the current champion's.

Semantic version rule: X.Y comes from name_that_llm.__version__. Raise it when the features or the model code
change (a new kind of model). Z goes up by one automatically for every retrain under the same X.Y.

Set MLFLOW_TRACKING_URI (for example http://mlflow:5000). Without it, runs go to a local ./mlruns folder.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from . import ROOT, __version__

MODEL_NAME = "name_that_llm"
EXPERIMENT = "name_that_llm"
CHAMPION = "champion"


def next_semver(client, name):
    """0.1.0, then 0.1.1 ... Z counts the versions already registered under the same X.Y."""
    base = ".".join(__version__.split(".")[:2])
    patches = []
    for version in client.search_model_versions(f"name='{name}'"):
        semver = version.tags.get("semver", "")
        if semver.startswith(base + "."):
            patches.append(int(semver.rsplit(".", 1)[1]))
    return f"{base}.{max(patches, default=-1) + 1}"


def metric_name(variant):
    """MLflow metric names can't contain "+": "style + embeddings" becomes "style_plus_embeddings"."""
    return variant.replace("+", "plus").replace(" ", "_").replace("__", "_")


def get_champion(client, name):
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None


def register_candidate(model_file, score, settings=None, extra_files=(), metrics=None, name=MODEL_NAME, min_gain=0.0):
    """Log a run, register the model as a new version, and make it the champion if it beats the current one.

    Returns a dict describing what happened.
    """
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    settings = settings or {}

    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params(settings)
        mlflow.log_metric("balanced_accuracy", score)
        for key, value in (metrics or {}).items():
            mlflow.log_metric(key, value)
        mlflow.log_artifact(str(model_file), artifact_path="model")
        for path in extra_files:
            mlflow.log_artifact(str(path), artifact_path="info")
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)

    semver = next_semver(client, name)
    client.set_model_version_tag(name, version.version, "semver", semver)
    client.set_model_version_tag(name, version.version, "balanced_accuracy", f"{score:.4f}")

    champion = get_champion(client, name)
    champion_score = float(champion.tags["balanced_accuracy"]) if champion is not None else None
    promoted = champion is None or score >= champion_score + min_gain
    if promoted:
        if champion is not None:
            client.set_model_version_tag(name, champion.version, "status", "retired")
        client.set_registered_model_alias(name, CHAMPION, version.version)
    client.set_model_version_tag(name, version.version, "status", "champion" if promoted else "not promoted")

    return {
        "version": int(version.version),
        "semver": semver,
        "score": score,
        "champion_score": champion_score,
        "promoted": promoted,
        "run_id": run.info.run_id,
    }


def load_champion(name=MODEL_NAME, version=None):
    """Download a registered model (the champion unless a version number is given) and load it.

    Returns (bundle, info) where info says which registry version was loaded.
    """
    client = MlflowClient()
    registered = client.get_model_version(name, str(version)) if version else get_champion(client, name)
    if registered is None:
        raise LookupError(f"no '{CHAMPION}' version of '{name}' in the registry yet")
    folder = mlflow.artifacts.download_artifacts(artifact_uri=registered.source)
    bundle = joblib.load(Path(folder) / "model.joblib")
    info = {"name": name, "version": int(registered.version), "semver": registered.tags.get("semver"),
            "balanced_accuracy": registered.tags.get("balanced_accuracy")}
    return bundle, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=ROOT / "artifacts" / "model.joblib")
    parser.add_argument("--metrics", default=ROOT / "artifacts" / "metrics.json", help="written by train --cv --metrics-out")
    parser.add_argument("--params", default=ROOT / "artifacts" / "best_params.json", help="written by tune (optional)")
    parser.add_argument("--name", default=MODEL_NAME)
    parser.add_argument("--min-gain", type=float, default=0.0, help="how much better than the champion a model must be")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    extra = [Path(p) for p in (args.metrics, args.params) if Path(p).exists()]
    result = register_candidate(
        args.model, metrics["balanced_accuracy"], metrics.get("settings"), extra,
        metrics={metric_name(variant) + "_mean": values["mean"] for variant, values in metrics["scores"].items()},
        name=args.name, min_gain=args.min_gain)

    print(f"registered {args.name} version {result['version']} (semver {result['semver']}), score {result['score']:.3f}")
    if result["promoted"]:
        print("-> now the champion")
    else:
        print(f"-> NOT promoted: champion scores {result['champion_score']:.3f}")
    print("tracking at", os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))


if __name__ == "__main__":
    main()
