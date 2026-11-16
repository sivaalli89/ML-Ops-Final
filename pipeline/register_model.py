"""Log the model the notebook just trained to MLflow, register it, and promote it if it is good enough.

Airflow runs this right after the notebook (dags/train_pipeline_dag.py). By hand:

    python register_model.py --models ../models --notebook ../runs/run_20261108T155737.ipynb

What it does, in order:
  1. Logs one MLflow run: the settings and scores as parameters and metrics, every Optuna trial as a nested run, and the
     files (model, model_info.json, held-out passages, trials, the executed notebook) as artifacts.
  2. Registers the model files as a new version of the registered model "name-that-llm".
  3. Gives it a semantic version. The notebook's MODEL_VERSION (for example 0.2.0) is used if no registered version has it
     yet; otherwise the last number goes up (0.2.1, 0.2.2 ...). Raise MODEL_VERSION in the notebook when the method changes.
  4. Promotes it: the alias "champion" moves to the new version only if its held-out score is at least as good as the
     current champion's. Otherwise the version is kept in the registry but the old champion keeps serving.

The API loads whatever "champion" points at, so promotion is what puts a model live.
Set MLFLOW_TRACKING_URI (Docker Compose does, for example http://mlflow:5000).
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "name-that-llm"
CHAMPION = "champion"


def next_semantic_version(client, name, base):
    """base is the notebook's MODEL_VERSION, for example "0.2.0". Use it if it is free, else count up the last number."""
    major_minor, first = base.rsplit(".", 1)
    used = {version.tags.get("semantic_version") for version in client.search_model_versions(f"name='{name}'")}
    patch = int(first)
    while f"{major_minor}.{patch}" in used:
        patch += 1
    return f"{major_minor}.{patch}"


def get_champion(client, name):
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None


def log_run(models, notebook, info, version):
    """One MLflow run for this training: parameters, metrics, the Optuna trials, and the files."""
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name=f"train {version}") as run:
        mlflow.log_params({**info["settings"], "optuna_trials": info["optuna_trials"], "training_passages": info["training_passages"],
                           "holdout_passages": info["holdout_passages"], "train_csv_md5": info["train_csv_md5"],
                           "notebook_version": info["version"]})
        mlflow.log_metrics({"holdout_balanced_accuracy": info["balanced_accuracy"], "cv_balanced_accuracy": info["cv_balanced_accuracy"]})
        mlflow.set_tag("semantic_version", version)

        trials_file = models / "optuna_trials.csv"
        for trial in pd.read_csv(trials_file).itertuples():
            with mlflow.start_run(run_name=f"optuna trial {trial.number}", nested=True):
                mlflow.log_params({"style_c": trial.params_style_c, "embedding_c": trial.params_embedding_c,
                                   "embedding_weight": round(trial.params_embedding_weight, 4)})
                mlflow.log_metric("cv_balanced_accuracy", trial.value)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "model"
            folder.mkdir()
            for name in ("model.joblib", "holdout.csv", "optuna_trials.csv"):
                shutil.copy(models / name, folder / name)
            # the copy that travels with the model says the registry's version number
            (folder / "model_info.json").write_text(json.dumps({**info, "version": version, "notebook_version": info["version"]}, indent=2))
            mlflow.log_artifacts(str(folder), artifact_path="model")
        if notebook and Path(notebook).exists():
            mlflow.log_artifact(str(notebook), artifact_path="executed_notebook")
    return run.info.run_id


def register(models, notebook=None, name=MODEL_NAME, min_gain=0.0):
    models = Path(models)
    info = json.loads((models / "model_info.json").read_text())
    score = info["balanced_accuracy"]
    client = MlflowClient()

    version = next_semantic_version(client, name, info["version"])
    run_id = log_run(models, notebook, info, version)

    registered = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.set_model_version_tag(name, registered.version, "semantic_version", version)
    client.set_model_version_tag(name, registered.version, "holdout_balanced_accuracy", f"{score:.4f}")
    client.set_model_version_tag(name, registered.version, "mlflow_run_id", run_id)

    champion = get_champion(client, name)
    champion_score = float(champion.tags["holdout_balanced_accuracy"]) if champion is not None else None
    promoted = champion is None or score >= champion_score + min_gain
    if promoted:
        if champion is not None:
            client.set_model_version_tag(name, champion.version, "status", "retired")
        client.set_registered_model_alias(name, CHAMPION, registered.version)
    client.set_model_version_tag(name, registered.version, "status", "champion" if promoted else "not promoted")

    return {"registry_version": int(registered.version), "semantic_version": version, "score": score,
            "champion_score": champion_score, "promoted": promoted, "run_id": run_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="../models", help="the folder the notebook saved the model in")
    parser.add_argument("--notebook", help="the executed copy of the notebook, kept with the run")
    parser.add_argument("--name", default=MODEL_NAME)
    parser.add_argument("--min-gain", type=float, default=0.0, help="how much better than the champion a model must score")
    args = parser.parse_args()

    result = register(args.models, args.notebook, args.name, args.min_gain)
    print(f"registered {args.name} registry version {result['registry_version']} = semantic version {result['semantic_version']}, "
          f"held-out score {result['score']:.3f}")
    if result["promoted"]:
        print("-> now the champion: the API will load it within a few seconds" if result["champion_score"] is not None else
              "-> now the champion (the first one): the API will load it within a few seconds")
    else:
        print(f"-> NOT promoted: the champion scores {result['champion_score']:.3f} and keeps serving")


if __name__ == "__main__":
    main()
