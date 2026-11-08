"""Model versioning against a throwaway MLflow store (a sqlite file in a temp folder)."""
import joblib
import pytest

pytest.importorskip("mlflow")
import mlflow  # noqa: E402

from name_that_llm.predict import predict_models  # noqa: E402
from name_that_llm.registry import get_champion, load_champion, register_candidate  # noqa: E402
from name_that_llm.train import fit_bundle  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # artifacts go to ./mlruns inside the temp folder
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield tmp_path


@pytest.fixture
def model_file(toy_data, store):
    bundle = fit_bundle(toy_data["text"], toy_data["model"], toy_data["prompt_id"].to_numpy())
    path = store / "model.joblib"
    joblib.dump(bundle, path)
    return path


def test_first_model_becomes_champion_with_a_semver(model_file):
    result = register_candidate(model_file, 0.50, {"style_c": 1.0})
    assert result["promoted"] is True
    assert result["version"] == 1
    assert result["semver"] == "0.1.0"
    assert int(get_champion(mlflow.MlflowClient(), "name_that_llm").version) == 1


def test_a_worse_model_is_registered_but_not_promoted(model_file):
    register_candidate(model_file, 0.50)
    worse = register_candidate(model_file, 0.45)
    client = mlflow.MlflowClient()
    assert worse["promoted"] is False
    assert worse["semver"] == "0.1.1"                 # still gets the next version number
    assert int(get_champion(client, "name_that_llm").version) == 1
    assert client.get_model_version("name_that_llm", "2").tags["status"] == "not promoted"


def test_a_better_model_takes_over_and_versions_keep_counting(model_file):
    register_candidate(model_file, 0.50)
    register_candidate(model_file, 0.45)
    better = register_candidate(model_file, 0.55)
    assert better["promoted"] is True
    assert better["semver"] == "0.1.2"
    assert int(get_champion(mlflow.MlflowClient(), "name_that_llm").version) == 3


def test_min_gain_blocks_a_tiny_improvement(model_file):
    register_candidate(model_file, 0.50)
    result = register_candidate(model_file, 0.501, min_gain=0.01)
    assert result["promoted"] is False


def test_champion_can_be_loaded_and_used(model_file, toy_data):
    register_candidate(model_file, 0.50)
    bundle, info = load_champion()
    assert info["version"] == 1 and info["semver"] == "0.1.0"
    prediction = predict_models(bundle, toy_data["text"][:8])
    assert len(prediction.models) == 8


def test_loading_before_anything_is_registered_fails_clearly(store):
    with pytest.raises(LookupError):
        load_champion()


def test_metric_names_from_the_training_scores_are_accepted_by_mlflow(model_file):
    from name_that_llm.registry import metric_name
    assert metric_name("style + embeddings") == "style_plus_embeddings"
    assert metric_name("style only") == "style_only"
    register_candidate(model_file, 0.5, metrics={metric_name("style + embeddings") + "_mean": 0.53})


def test_the_old_champion_is_marked_retired(model_file):
    register_candidate(model_file, 0.50)
    register_candidate(model_file, 0.55)
    client = mlflow.MlflowClient()
    assert client.get_model_version("name_that_llm", "1").tags["status"] == "retired"
    assert client.get_model_version("name_that_llm", "2").tags["status"] == "champion"
