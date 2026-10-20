import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path, monkeypatch):
  monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
  monkeypatch.chdir(tmp_path)


def _make_df():
  rng = np.random.default_rng(0)
  n = 120
  df = pd.DataFrame({
    "age": rng.integers(30, 75, n), "sex": rng.integers(0, 2, n), "cp": rng.integers(1, 5, n),
    "trestbps": rng.integers(100, 180, n), "chol": rng.integers(150, 350, n),
    "fbs": rng.integers(0, 2, n), "restecg": rng.integers(0, 3, n),
    "thalach": rng.integers(90, 200, n), "exang": rng.integers(0, 2, n),
    "oldpeak": rng.random(n) * 4, "slope": rng.integers(1, 4, n),
    "ca": rng.integers(0, 4, n).astype(float), "thal": rng.choice([3.0, 6.0, 7.0], n),
  })
  df["target"] = (df["oldpeak"] + df["exang"] + rng.normal(0, 0.5, n) > 2).astype(int)
  return df


@pytest.fixture
def synthetic_csv(tmp_path):
  p = tmp_path / "heart.csv"
  _make_df().to_csv(p, index=False)
  return str(p)


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
  """Train once per test session on synthetic data (MLflow logged into a temp dir)."""
  import os

  from heart.train import train

  d = tmp_path_factory.mktemp("trained")
  csv = d / "heart.csv"
  _make_df().to_csv(csv, index=False)
  os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(d / 'mlflow.db').as_posix()}"
  try:
    train(str(csv), str(d / "m.joblib"), experiment="test", cv=3)
  finally:
    del os.environ["MLFLOW_TRACKING_URI"]
  return str(d / "m.joblib")
