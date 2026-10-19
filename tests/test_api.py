import pytest
from fastapi.testclient import TestClient

GOOD = dict(age=55, sex=1, cp=3, trestbps=130, chol=240, fbs=0, restecg=0,
            thalach=150, exang=1, oldpeak=2.0, slope=2, ca=0, thal=3)


@pytest.fixture
def client(trained_model_path, monkeypatch):
  monkeypatch.setenv("MODEL_PATH", trained_model_path)
  from api.app import create_app
  with TestClient(create_app()) as c:
    yield c


def test_predict_ok(client):
  r = client.post("/predict", json=GOOD)
  assert r.status_code == 200
  assert set(r.json()) == {"prediction", "confidence"}


def test_missing_ca_thal_accepted(client):
  body = {k: v for k, v in GOOD.items() if k not in ("ca", "thal")}
  assert client.post("/predict", json=body).status_code == 200


def test_rejects_bad_values(client):
  assert client.post("/predict", json={**GOOD, "age": -5}).status_code == 422
  assert client.post("/predict", json={**GOOD, "cp": 9}).status_code == 422
  assert client.post("/predict", json={**GOOD, "thal": 5}).status_code == 422
  missing_age = {k: v for k, v in GOOD.items() if k != "age"}
  assert client.post("/predict", json=missing_age).status_code == 422


def test_extra_fields_ignored(client):
  assert client.post("/predict", json={**GOOD, "foo": 1}).status_code == 200


def test_health_and_metrics(client):
  client.post("/predict", json=GOOD)
  assert client.get("/health").json() == {"status": "ok"}
  assert "predict_requests_total" in client.get("/metrics").text


def test_missing_model_fails_fast(tmp_path, monkeypatch):
  monkeypatch.setenv("MODEL_PATH", str(tmp_path / "none.joblib"))
  from api.app import create_app
  with pytest.raises(FileNotFoundError):
    with TestClient(create_app()):
      pass


def test_error_and_ok_series_exist_before_any_request(client):
  text = client.get("/metrics").text
  assert 'predict_requests_total{outcome="error"}' in text
