import joblib
import pandas as pd
import pytest

from heart.train import train


def test_train_saves_pipeline_and_reports_metrics(synthetic_csv, tmp_path):
  out = train(synthetic_csv, str(tmp_path / "m.joblib"), experiment="test", cv=3)
  assert out["metrics"]["roc_auc"] > 0.5
  model = joblib.load(tmp_path / "m.joblib")
  raw = pd.read_csv(synthetic_csv).drop(columns="target").head(2)
  assert model.predict_proba(raw).shape == (2, 2)


def test_train_rejects_single_class(synthetic_csv, tmp_path):
  df = pd.read_csv(synthetic_csv)
  df["target"] = 1
  df.to_csv(synthetic_csv, index=False)
  with pytest.raises(ValueError, match="both classes"):
    train(synthetic_csv, str(tmp_path / "m.joblib"), experiment="test", cv=3)
