"""HTTP service: send a batch of passages, get a guess of the writing model for each one.

    uvicorn api.main:app --port 8000

Settings (environment variables):
    MLFLOW_TRACKING_URI + MODEL_NAME   load the "champion" version from the MLflow Model Registry
    MODEL_VERSION    with the two above: load this registry version instead of the champion
    MODEL_PATH       saved model file, used when the registry is not set up or has no champion yet
    USE_EMBEDDINGS   "true" to blend in DistilBERT (needs PyTorch + sentence-transformers), default false
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from name_that_llm.grouping import MIN_BATCH
from name_that_llm.predict import predict_models

MAX_PASSAGES = 5000
state = {}


def load_model():
    """Registry champion if the registry is set up, otherwise the saved file. Returns (bundle, info)."""
    if os.environ.get("MLFLOW_TRACKING_URI") and os.environ.get("MODEL_NAME"):
        try:
            from name_that_llm.registry import load_champion
            bundle, info = load_champion(os.environ["MODEL_NAME"], os.environ.get("MODEL_VERSION") or None)
            return bundle, {"source": "registry", **info}
        except Exception as error:   # registry down, or nothing registered yet
            print(f"could not load from the registry ({error}); trying the model file", flush=True)

    model_path = Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
    if model_path.exists():
        return joblib.load(model_path), {"source": "file", "path": str(model_path)}
    return None, None


@asynccontextmanager
async def lifespan(app):
    state["bundle"], state["model_info"] = load_model()
    state["use_embeddings"] = os.environ.get("USE_EMBEDDINGS", "false").lower() == "true"
    yield
    state.clear()


app = FastAPI(title="Name That LLM", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    passages: List[str] = Field(min_length=1, max_length=MAX_PASSAGES)

    @field_validator("passages")
    @classmethod
    def no_empty_passages(cls, passages):
        if any(not p.strip() for p in passages):
            raise ValueError("passages must not be empty")
        return passages


class PredictResponse(BaseModel):
    mode: str                     # "grouped" or "ungrouped"
    models: List[str]             # one guess per passage, in the same order
    groups: Optional[List[int]]   # which prompt group each passage was put in
    note: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.get("bundle") is not None,
        "model": state.get("model_info"),
        "embeddings": state.get("use_embeddings", False),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    prediction = predict_models(bundle, request.passages, use_embeddings=state["use_embeddings"])
    note = None
    if prediction.mode == "ungrouped":
        note = (f"Batch smaller than {MIN_BATCH} passages: passages could not be grouped by prompt, "
                "so the weaker plain classifier was used.")
    groups = None if prediction.groups is None else [int(g) for g in prediction.groups]
    return PredictResponse(mode=prediction.mode, models=prediction.models, groups=groups, note=note)
