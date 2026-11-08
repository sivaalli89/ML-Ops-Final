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
