import numpy as np
import pandas as pd

from heart.features import add_features, build_preprocessor


def _row(**kw):
  base = dict(age=50, sex=1, cp=3, trestbps=120, chol=200, fbs=0, restecg=0,
              thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0.0, thal=3.0)
  base.update(kw)
  return pd.DataFrame([base])


def test_add_features_hr_reserve():
  assert add_features(_row())["hr_reserve"].iloc[0] == (220 - 50) - 150


def test_preprocessor_handles_missing_and_unseen():
  X = pd.concat([_row(), _row(age=60, cp=1)], ignore_index=True)
  pre = build_preprocessor().fit(add_features(X))
  out = pre.transform(add_features(_row(ca=np.nan, thal=np.nan, cp=99)))
  assert not np.isnan(out).any()
