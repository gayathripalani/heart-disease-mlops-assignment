import pytest

from heart.predict import load_model, predict_one

REC = dict(age=55, sex=1, cp=3, trestbps=130, chol=240, fbs=0, restecg=0,
           thalach=150, exang=1, oldpeak=2.0, slope=2, ca=0.0, thal=3.0)


def test_predict_one(trained_model_path):
  out = predict_one(load_model(trained_model_path), REC)
  assert out["prediction"] in (0, 1)
  assert 0.5 <= out["confidence"] <= 1


def test_missing_model_message(tmp_path):
  with pytest.raises(FileNotFoundError, match="model"):
    load_model(str(tmp_path / "nope.joblib"))
