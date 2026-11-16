"""A small prediction service for the Name That LLM model, with monitoring.

Send it the passages that answer ONE prompt (ideally one from each LLM) and it says which LLM wrote each one.

    POST /predict     {"passages": ["...", "..."]}  ->  one guessed LLM per passage
    GET  /model       which model is loaded, where it came from, how well it did on held-out prompts
    GET  /monitor     drift report as JSON (status OK, ALERT or WAITING)
    GET  /dashboard   the same report as a web page; /dashboard/evidently is Evidently's own full report
    GET  /health

Where the model comes from:
  * MLFLOW_TRACKING_URI set (Docker Compose does this): the "champion" version in the MLflow model registry. Every
    CHECK_EVERY seconds a request looks at the registry, and if the champion changed (a new model was promoted, or
    someone rolled back) the service loads it. No restart is needed.
  * not set: the files in models/ (written by the notebook). The service notices when model_info.json changes.
  If the new model can't be loaded, the service keeps answering with the one it has.

Monitoring: see monitoring.py in this folder (Evidently data drift reports). When a model is loaded, the held-out passages that were saved with it are run through the
same code as a live request, which gives the baseline. Every live request is compared with that baseline.

If sentence-transformers is installed, the DistilBERT part of the ensemble is used too; if not, only the style model
answers (a little less accurate). Every answer says which one was used.

Run (from the project folder):  uvicorn service.api:app --port 8000      then open http://localhost:8000/docs
"""
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from service import monitoring

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
MODEL_NAME = os.environ.get("MODEL_NAME", "name-that-llm")
CHECK_EVERY = 10          # seconds between looks at the registry
MONITOR_DIR = os.environ.get("MONITOR_DIR", "monitoring")


# ---- the model: the same helper functions as the notebook, written for one prompt's passages ----

def make_skeleton(text, common_words):
    pieces = []
    for token in re.findall(r"[A-Za-z']+|[^\s A-Za-z']", text):
        lower = token.lower()
        if lower in common_words:
            pieces.append(lower)
        elif token[0].isalpha() or token[0] == "'":
            pieces.append("W")
        else:
            pieces.append(token)
    return " ".join(pieces)


def transform_features(features, texts):
    blocks = []
    for name, vec in features["vectorizers"].items():
        docs = [make_skeleton(t, features["common_words"]) for t in texts] if name == "skeleton" else texts
        blocks.append(vec.transform(docs).toarray())
    return np.hstack(blocks)


def center(features):
    # all the passages are from one prompt, so subtract their average
    return features - features.mean(axis=0)


def blend(probs_main, probs_other, other_weight):
    logs = (1 - other_weight) * np.log(probs_main + 1e-9) + other_weight * np.log(probs_other + 1e-9)
    mixed = np.exp(logs)
    return mixed / mixed.sum(axis=1, keepdims=True)


def assign_one_model_each(probs, n_models):
    cost = -np.log(probs + 1e-9)
    if len(probs) > n_models:                        # more passages than models: let models repeat
        cost = np.tile(cost, int(np.ceil(len(probs) / n_models)))
    guess = np.empty(len(probs), dtype=int)
    row_ids, model_ids = linear_sum_assignment(cost)
    guess[row_ids] = model_ids % n_models
    return guess


def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    embedder = SentenceTransformer("distilbert-base-uncased-finetuned-sst-2-english", device="cpu")
    embedder.max_seq_length = 512
    return embedder


embedder = load_embedder()


@dataclass
class Loaded:
    saved: dict
    info: dict
    source: str                  # where it came from, for people
    stamp: str                   # what identifies it, to notice a change
    vocabulary: set
    baseline: pd.DataFrame = None
    baseline_accuracy: float = None
    version: str = field(init=False)

    def __post_init__(self):
        self.version = str(self.info["version"])


def probabilities(loaded, texts):
    saved = loaded.saved
    probs = saved["style_classifier"].predict_proba(center(transform_features(saved["features"], texts)))
    if embedder is not None:
        embeddings = embedder.encode(texts, batch_size=16, normalize_embeddings=True)
        embedding_x = saved["embedding_scaler"].transform(center(embeddings))
        embedding_probs = saved["embedding_classifier"].predict_proba(embedding_x)
        probs = blend(probs, embedding_probs, saved["embedding_weight"])
    return probs


def guess_llms(loaded, probs):
    classes = loaded.saved["classes"]
    return [str(name) for name in classes[assign_one_model_each(probs, len(classes))]]


def compute_baseline(loaded, holdout):
    """Run the held-out prompts through the service's own code. The accuracy uses all of them; the baseline numbers that
    live requests are compared with use only the baseline half (the other half is kept as a clean test, see service/monitoring.py)."""
    baseline_groups = set(monitoring.split_holdout(holdout)[0]["group"])
    frames, right, total = [], 0, 0
    for group_id, group in holdout.groupby("group"):
        texts = group["text"].tolist()
        if len(texts) < 2:
            continue
        probs = probabilities(loaded, texts)
        right += sum(a == b for a, b in zip(guess_llms(loaded, probs), group["model"]))
        total += len(texts)
        if group_id in baseline_groups:
            frames.append(monitoring.request_stats(loaded.saved, texts, probs, loaded.vocabulary))
    return pd.concat(frames, ignore_index=True), round(right / total, 3)


def read_model_folder(folder, source, stamp):
    saved = joblib.load(Path(folder) / "model.joblib")
    info = json.loads((Path(folder) / "model_info.json").read_text())
    vocabulary = set(saved["features"]["vectorizers"]["words"].vocabulary_)      # the words and word pairs the model knows
    loaded = Loaded(saved, info, source, stamp, vocabulary)
    holdout_file = Path(folder) / "holdout.csv"
    if holdout_file.exists():
        loaded.baseline, loaded.baseline_accuracy = compute_baseline(loaded, pd.read_csv(holdout_file))
    print("loaded model", loaded.version, "from", source, flush=True)
    return loaded


class ModelHolder:
    """Holds the model that is answering, and swaps it when a newer one appears."""

    def __init__(self):
        self.loaded = None
        self.last_check = 0.0
        self.lock = threading.Lock()
        self.cache = Path(tempfile.gettempdir()) / "model-versions"
        if MLFLOW_URI:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
        self.refresh()

    def get(self):
        if MLFLOW_URI and time.time() - self.last_check > CHECK_EVERY:
            self.refresh()
        elif not MLFLOW_URI and self._local_stamp() != getattr(self.loaded, "stamp", None):
            self.refresh()
        return self.loaded

    def _local_stamp(self):
        info_file = MODEL_DIR / "model_info.json"
        return str(info_file.stat().st_mtime_ns) if info_file.exists() else None

    def refresh(self):
        if not self.lock.acquire(blocking=False):        # someone else is already loading: keep using the current model
            return
        try:
            self.last_check = time.time()
            if MLFLOW_URI:
                self._refresh_from_registry()
            else:
                self._refresh_from_files()
        except Exception as error:                        # half-written files, registry down ...: keep the old model
            print("could not load a new model, keeping the current one:", error, flush=True)
        finally:
            self.lock.release()

    def _refresh_from_files(self):
        stamp = self._local_stamp()                       # read before loading, so a change during loading is seen next time
        if stamp is not None and stamp != getattr(self.loaded, "stamp", None):
            self.loaded = read_model_folder(MODEL_DIR, f"files in {MODEL_DIR}", stamp)

    def _refresh_from_registry(self):
        import mlflow
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        try:
            champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        except MlflowException:
            return                                        # nothing promoted yet
        if getattr(self.loaded, "stamp", None) != champion.version:
            target = self.cache / f"v{champion.version}"
            target.mkdir(parents=True, exist_ok=True)         # MLflow won't create it
            folder = mlflow.artifacts.download_artifacts(artifact_uri=champion.source, dst_path=str(target))
            self.loaded = read_model_folder(folder, f"MLflow registry, champion = version {champion.version}", champion.version)


holder = ModelHolder()
monitor = monitoring.Monitor(MONITOR_DIR)


# ---- the web service ----

class PredictRequest(BaseModel):
    passages: list[str] = Field(min_length=2, max_length=200, description="passages that answer the same prompt")


class Answer(BaseModel):
    model_version: str
    used_embeddings: bool
    llms: list[str] = Field(description="the guessed LLM for each passage, in the same order")


app = FastAPI(title="Name That LLM")


@app.exception_handler(RequestValidationError)
async def count_rejected_requests(request: Request, error: RequestValidationError):
    if request.url.path == "/predict":
        monitor.record_invalid()                           # a malformed request is something to watch
    return await request_validation_exception_handler(request, error)


def current_model():
    loaded = holder.get()
    if loaded is None:
        raise HTTPException(503, "No model yet: train one (run the notebook or the Airflow DAG) and it will be picked up.")
    return loaded


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": holder.get() is not None}       # get() also looks for a new champion, at most every 10 seconds


@app.get("/model")
def model_info():
    loaded = current_model()
    return {**loaded.info, "source": loaded.source, "served_holdout_accuracy": loaded.baseline_accuracy}


@app.post("/predict", response_model=Answer)
def predict_llms(request: PredictRequest):
    loaded = current_model()
    probs = probabilities(loaded, request.passages)
    monitor.record(loaded.version, monitoring.request_stats(loaded.saved, request.passages, probs, loaded.vocabulary))
    return Answer(model_version=loaded.version, used_embeddings=embedder is not None, llms=guess_llms(loaded, probs))


@app.get("/monitor")
def monitor_report(last: int = 500, last_requests: int = 100):
    """Drift report: the last `last` passages against the held-out baseline, and the rejected requests among the last `last_requests`."""
    loaded = current_model()
    return monitor.report(loaded.baseline, loaded.version, last, last_requests)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    loaded = current_model()
    report = monitor.report(loaded.baseline, loaded.version)
    return monitoring.render_dashboard(report, {"version": loaded.version, "source": loaded.source, "baseline_accuracy": loaded.baseline_accuracy})


@app.get("/dashboard/evidently", response_class=HTMLResponse)
def evidently_report(last: int = 500):
    """Evidently's full data drift report (a distribution plot per column) for the last `last` passages."""
    loaded = current_model()
    return monitor.evidently_html(loaded.baseline, loaded.version, last)
