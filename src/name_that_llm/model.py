"""The trained model: fitted features plus the two classifiers, and how they are blended.

Kept in its own module so a saved model can be loaded from anywhere
(a class defined in a script run with -m would be pickled as __main__.ModelBundle).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import StyleFeatures
from .group_ops import center_by_group

EMBEDDING_WEIGHT = 0.1   # DistilBERT is weaker than TF-IDF alone; at 0.3 and above it starts to hurt


def blend(probs_main, probs_other, other_weight):
    """Weighted geometric mean: average the log-probabilities, then turn them back into probabilities."""
    logs = (1 - other_weight) * np.log(probs_main + 1e-9) + other_weight * np.log(probs_other + 1e-9)
    mixed = np.exp(logs)
    return mixed / mixed.sum(axis=1, keepdims=True)


@dataclass
class ModelBundle:
    """Everything needed to predict: the fitted features and both classifiers."""
    classes: np.ndarray
    style: StyleFeatures
    style_model: LogisticRegression
    embedding_scaler: StandardScaler = None
    embedding_model: LogisticRegression = None
    embedding_weight: float = EMBEDDING_WEIGHT

    def predict_parts(self, texts, groups=None, embeddings=None):
        """Probabilities from the style model and (if available) the embedding model."""
        style_x = self.style.transform(texts)
        if groups is not None:
            style_x = center_by_group(style_x, groups)
        style_probs = self.style_model.predict_proba(style_x)

        embedding_probs = None
        if self.embedding_model is not None and embeddings is not None:
            if groups is not None:
                embeddings = center_by_group(embeddings, groups)
            embedding_probs = self.embedding_model.predict_proba(self.embedding_scaler.transform(embeddings))
        return style_probs, embedding_probs

    def predict_proba(self, texts, groups=None, embeddings=None):
        style_probs, embedding_probs = self.predict_parts(texts, groups, embeddings)
        if embedding_probs is None:
            return style_probs
        return blend(style_probs, embedding_probs, self.embedding_weight)
