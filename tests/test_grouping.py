from name_that_llm.grouping import cluster_passages, cluster_quality
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
