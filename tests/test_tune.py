import json

import numpy as np
import pytest

pytest.importorskip("optuna")

from name_that_llm.train import cross_validate, load_params  # noqa: E402
from name_that_llm.tune import nested_cv, tune  # noqa: E402


@pytest.fixture
def toy_embeddings(toy_data):
    # made-up "language model" numbers that carry a faint model signal
    rng = np.random.default_rng(3)
    codes = toy_data["model"].str[-1].astype(int).to_numpy()
    return rng.normal(size=(len(toy_data), 8)) + np.eye(4)[codes].repeat(2, axis=1)


def test_tuning_returns_settings_inside_the_search_ranges(toy_data, toy_embeddings):
    result = tune(toy_data["text"], toy_data["model"], toy_data["prompt_id"].to_numpy(), toy_embeddings,
                  n_trials=3, n_splits=3, n_jobs=2, verbose=False)
    params = result["best_params"]
    assert 0.3 <= params["style_c"] <= 10
    assert 0.01 <= params["embedding_c"] <= 1
    assert 0 <= params["embedding_weight"] <= 0.3
    assert result["cv_score_tuned"] >= result["cv_score_default"]   # trial 0 is the default, best can't be lower


def test_without_embeddings_only_the_style_setting_is_tuned(toy_data):
    result = tune(toy_data["text"], toy_data["model"], toy_data["prompt_id"].to_numpy(), None,
                  n_trials=2, n_splits=3, n_jobs=2, verbose=False)
    assert list(result["best_params"]) == ["style_c"]


def test_tuned_settings_can_be_saved_and_used_for_training(toy_data, tmp_path):
    path = tmp_path / "best.json"
    path.write_text(json.dumps({"best_params": {"style_c": 2.0, "unrelated": 1}, "cv_score_tuned": 0.5}))
    params = load_params(path)
    assert params == {"style_c": 2.0}                # unknown keys are ignored
    scores = cross_validate(toy_data["text"], toy_data["model"], toy_data["prompt_id"].to_numpy(), n_jobs=1, params=params)
    assert len(scores) == 5


def test_nested_check_scores_default_and_tuned_on_untouched_prompts(toy_data):
    scores = nested_cv(toy_data["text"], toy_data["model"], toy_data["prompt_id"].to_numpy(), None,
                       n_trials=2, n_outer=2, n_inner=2, n_jobs=1)
    assert list(scores.columns) == ["default", "tuned"]
    assert len(scores) == 2
    assert scores.notna().all().all()
