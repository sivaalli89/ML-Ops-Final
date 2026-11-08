"""HTTP service: send a batch of passages, get a guess of the writing model for each one.

    uvicorn api.main:app --port 8000

Settings (environment variables):
    MODEL_PATH       saved model, default artifacts/model.joblib
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


@asynccontextmanager
async def lifespan(app):
    model_path = Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
    state["bundle"] = joblib.load(model_path) if model_path.exists() else None
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
