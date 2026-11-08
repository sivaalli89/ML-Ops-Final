"""Guess the model for a batch of passages."""
from dataclasses import dataclass

import numpy as np

from .features import embed_passages
from .group_ops import assign_one_model_each
from .grouping import MIN_BATCH, cluster_passages
from .model import ModelBundle


@dataclass
class Prediction:
    models: list         # one model name per passage
    groups: np.ndarray   # prompt group of each passage (None if the batch was too small to group)
    mode: str            # "grouped" (centering + assignment) or "ungrouped" (plain classifier)
    probs: np.ndarray


def predict_models(bundle: ModelBundle, texts, embeddings=None, groups=None):
    """Centering and assignment need the other passages from the same prompts.

    So we cluster the batch first. A batch smaller than MIN_BATCH cannot be grouped, and falls back
    to the plain classifier, which is weaker.
    """
    texts = list(texts)
    if embeddings is None and bundle.embedding_model is not None:
        embeddings = embed_passages(texts)
    if groups is None and len(texts) >= MIN_BATCH:
        groups = cluster_passages(texts, n_models=len(bundle.classes))

    probs = bundle.predict_proba(texts, groups, embeddings)
    if groups is None:
        guess, mode = probs.argmax(axis=1), "ungrouped"
    else:
        guess, mode = assign_one_model_each(probs, groups), "grouped"
    return Prediction(list(bundle.classes[guess]), groups, mode, probs)
