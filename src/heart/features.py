import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.data import CATEGORICAL, NUMERIC

ENGINEERED = ["hr_reserve", "chol_age_ratio"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()
  out["hr_reserve"] = (220 - out["age"]) - out["thalach"]
  out["chol_age_ratio"] = out["chol"] / out["age"]
  return out


def build_preprocessor() -> ColumnTransformer:
  num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
  cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                  ("oh", OneHotEncoder(handle_unknown="ignore"))])
  return ColumnTransformer([("num", num, NUMERIC + ENGINEERED), ("cat", cat, CATEGORICAL)])
