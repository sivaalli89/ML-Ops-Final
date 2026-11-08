"""The two tricks that use the prompt groups: centering and one-model-each assignment."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def center_by_group(features, groups):
    """Subtract each group's average row. All passages in a group share the topic, so it cancels out."""
    centered = features.copy()
    for g in np.unique(groups):
        rows = (groups == g)
        centered[rows] = features[rows] - features[rows].mean(axis=0)
    return centered


def assign_one_model_each(probs, groups):
    """Inside a group each model should appear once, so find the best one-to-one matching."""
    n_models = probs.shape[1]
    best_guess = probs.argmax(axis=1)
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        cost = -np.log(probs[rows] + 1e-9)          # low cost = likely
        if len(rows) > n_models:                     # a merged cluster: let models repeat
            cost = np.tile(cost, int(np.ceil(len(rows) / n_models)))
        row_ids, model_ids = linear_sum_assignment(cost)
        best_guess[rows[row_ids]] = model_ids % n_models
    return best_guess
