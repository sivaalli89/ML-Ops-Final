"""End to end on the toy data: train, save, load, predict."""
import joblib
import numpy as np

from name_that_llm.grouping import MIN_BATCH
from name_that_llm.predict import predict_models
from name_that_llm.train import fit_bundle
from conftest import N_MODELS


def _train(toy_data):
    groups = toy_data["prompt_id"].to_numpy()
    return fit_bundle(toy_data["text"], toy_data["model"], groups)


def test_trained_model_recognises_the_style_habits(toy_data):
    bundle = _train(toy_data)
    assert len(bundle.classes) == N_MODELS
    prediction = predict_models(bundle, toy_data["text"], groups=toy_data["prompt_id"].to_numpy())
    accuracy = np.mean(np.array(prediction.models) == toy_data["model"].to_numpy())
    assert prediction.mode == "grouped"
    assert accuracy > 0.9


def test_a_big_batch_is_grouped_by_itself(toy_data):
    assert len(toy_data) >= MIN_BATCH
    prediction = predict_models(_train(toy_data), toy_data["text"])
    assert prediction.mode == "grouped"
    assert prediction.groups is not None
    assert len(prediction.models) == len(toy_data)


def test_a_small_batch_falls_back_to_the_plain_classifier(toy_data):
    prediction = predict_models(_train(toy_data), toy_data["text"][:8])
    assert prediction.mode == "ungrouped"
    assert prediction.groups is None
    assert len(prediction.models) == 8


def test_saved_model_loads_and_gives_the_same_answers(toy_data, tmp_path):
    bundle = _train(toy_data)
    joblib.dump(bundle, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    texts = toy_data["text"][:8]
    assert predict_models(loaded, texts).models == predict_models(bundle, texts).models
