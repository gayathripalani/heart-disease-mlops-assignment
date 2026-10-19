import io
import os
import urllib.request
import zipfile

import pandas as pd

# Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease
URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
MEMBER = "processed.cleveland.data"
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def download(dest: str) -> str:
  if os.path.exists(dest):
    return dest
  os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
  try:
    with urllib.request.urlopen(URL, timeout=30) as r:
      archive = zipfile.ZipFile(io.BytesIO(r.read()))
    data = archive.read(MEMBER)
  except (OSError, zipfile.BadZipFile, KeyError) as e:
    raise RuntimeError(f"Could not download dataset from {URL}: {e!r}") from e
  tmp = dest + ".tmp"
  with open(tmp, "wb") as out:
    out.write(data)
  os.replace(tmp, dest)
  return dest


def load_raw(path: str) -> pd.DataFrame:
  return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
  out = df.drop_duplicates().copy()
  out[TARGET] = (out["num"] > 0).astype(int)
  return out.drop(columns="num").reset_index(drop=True)
