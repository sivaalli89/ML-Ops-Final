import joblib
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from name_that_llm.train import fit_bundle  # noqa: E402


@pytest.fixture
def client(toy_data, tmp_path, monkeypatch):
    bundle = fit_bundle(toy_data["text"], toy_data["model"], toy_data["prompt_id"].to_numpy())
    joblib.dump(bundle, tmp_path / "model.joblib")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    from api.main import app
    with TestClient(app) as test_client:      # the "with" runs the startup code that loads the model
        yield test_client


def test_health_says_the_model_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_big_batch_is_grouped(client, toy_data):
    response = client.post("/predict", json={"passages": toy_data["text"].tolist()})
    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "grouped"
    assert len(body["models"]) == len(toy_data)
    assert len(body["groups"]) == len(toy_data)
    assert body["note"] is None


def test_small_batch_comes_with_a_warning(client, toy_data):
    body = client.post("/predict", json={"passages": toy_data["text"][:5].tolist()}).json()
    assert body["mode"] == "ungrouped"
    assert body["groups"] is None
    assert "smaller than" in body["note"]


@pytest.mark.parametrize("passages", [[], [""], ["fine text", "   "]])
def test_bad_input_is_rejected(client, passages):
    assert client.post("/predict", json={"passages": passages}).status_code == 422


def test_missing_model_gives_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nothing.joblib"))
    from api.main import app
    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        assert test_client.post("/predict", json={"passages": ["some text"]}).status_code == 503


def test_api_loads_the_champion_from_the_registry(toy_data, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow
    from name_that_llm.registry import register_candidate

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MODEL_NAME", "name_that_llm")
    mlflow.set_tracking_uri(uri)
    bundle = fit_bundle(toy_data["text"], toy_data["model"], toy_data["prompt_id"].to_numpy())
    joblib.dump(bundle, tmp_path / "model.joblib")
    register_candidate(tmp_path / "model.joblib", 0.5)

    from api.main import app
    with TestClient(app) as test_client:
        body = test_client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model"]["source"] == "registry"
    assert body["model"]["semver"] == "0.1.0"


def test_api_falls_back_to_the_file_when_the_registry_is_empty(toy_data, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow

    uri = f"sqlite:///{tmp_path / 'empty.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MODEL_NAME", "name_that_llm")
    mlflow.set_tracking_uri(uri)
    bundle = fit_bundle(toy_data["text"], toy_data["model"], toy_data["prompt_id"].to_numpy())
    joblib.dump(bundle, tmp_path / "model.joblib")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))

    from api.main import app
    with TestClient(app) as test_client:
        body = test_client.get("/health").json()
    assert body["model"]["source"] == "file"
