import numpy as np

from name_that_llm.grouping import assign_one_model_each, center_by_group, cluster_passages, cluster_quality
from conftest import N_MODELS


def test_clustering_recovers_the_prompts(toy_data):
    groups = cluster_passages(toy_data["text"], n_models=N_MODELS)
    quality = cluster_quality(groups, toy_data["model"].to_numpy(), toy_data["prompt_id"].to_numpy())
    assert quality["n_groups"] == 20
    assert quality["ari"] > 0.95
    assert quality["share_with_repeat"] == 0
    assert quality["median_size"] == N_MODELS


def test_cluster_quality_flags_a_repeated_model():
    quality = cluster_quality([0, 0, 1, 1], models=["a", "a", "a", "b"])
    assert quality["share_with_repeat"] == 0.5


def test_centering_makes_every_group_average_zero():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(12, 5)) + 10
    groups = np.repeat([0, 1, 2], 4)
    centered = center_by_group(features, groups)
    for g in range(3):
        assert np.allclose(centered[groups == g].mean(axis=0), 0)


def test_centering_does_not_change_the_input():
    features = np.arange(8.0).reshape(4, 2)
    center_by_group(features, np.array([0, 0, 1, 1]))
    assert features[0, 0] == 0.0


def test_assignment_gives_each_model_once_per_group():
    # every passage "wants" model 0 most, but a group of 4 must use all 4 models
    probs = np.tile([0.7, 0.1, 0.1, 0.1], (4, 1))
    guess = assign_one_model_each(probs, np.zeros(4, dtype=int))
    assert sorted(guess) == [0, 1, 2, 3]


def test_assignment_keeps_the_confident_guess():
    probs = np.array([[0.97, 0.01, 0.01, 0.01],
                      [0.4, 0.3, 0.2, 0.1],
                      [0.3, 0.4, 0.2, 0.1],
                      [0.25, 0.25, 0.25, 0.25]])
    guess = assign_one_model_each(probs, np.zeros(4, dtype=int))
    assert guess[0] == 0
    assert len(set(guess)) == 4


def test_assignment_lets_models_repeat_in_a_merged_group():
    probs = np.full((6, 3), 1 / 3)       # 6 passages, only 3 models: a merged cluster
    guess = assign_one_model_each(probs, np.zeros(6, dtype=int))
    assert len(guess) == 6
    assert set(guess) <= {0, 1, 2}
