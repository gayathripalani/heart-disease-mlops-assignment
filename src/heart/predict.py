import os

import joblib
import pandas as pd


def load_model(path=None):
  path = path or os.getenv("MODEL_PATH", "models/model.joblib")
  if not os.path.exists(path):
    raise FileNotFoundError(f"model file not found at {path}; run `python -m heart.train`")
  return joblib.load(path)


def predict_one(model, record: dict) -> dict:
  proba = model.predict_proba(pd.DataFrame([record]))[0]
  pred = int(proba.argmax())
  return {"prediction": pred, "confidence": float(proba[pred])}
