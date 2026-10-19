"""Train, tune and evaluate candidate models; log everything to MLflow; save the best."""
import os
import tempfile

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
  GridSearchCV,
  StratifiedKFold,
  cross_val_predict,
  cross_validate,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

from heart.data import TARGET  # noqa: E402
from heart.features import add_features, build_preprocessor  # noqa: E402

SCORING = ["accuracy", "precision", "recall", "roc_auc"]
MODELS = {
  "logreg": (LogisticRegression(max_iter=1000), {"clf__C": [0.01, 0.1, 1, 10]}),
  "random_forest": (RandomForestClassifier(random_state=42),
                    {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None]}),
  "gradient_boosting": (GradientBoostingClassifier(random_state=42),
                        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]}),
}


def _pipeline(clf):
  return Pipeline([("features", FunctionTransformer(add_features)),
                   ("pre", build_preprocessor()), ("clf", clf)])


def _log_plots(pipe, X, y, folds, name):
  """Honest (out-of-fold) ROC curve and confusion matrix, logged as run artifacts."""
  proba = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba")[:, 1]
  with tempfile.TemporaryDirectory() as tmp:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=name)
    ax.set_title(f"Out-of-fold ROC: {name}")
    fig.savefig(os.path.join(tmp, f"roc_{name}.png"))
    plt.close(fig)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, (proba >= 0.5).astype(int), ax=ax)
    ax.set_title(f"Confusion matrix: {name}")
    fig.savefig(os.path.join(tmp, f"cm_{name}.png"))
    plt.close(fig)
    mlflow.log_artifacts(tmp, artifact_path="plots")


def train(data_path, model_path="models/model.joblib", experiment="heart-disease", cv=5):
  df = pd.read_csv(data_path)
  y = df.pop(TARGET)
  if y.nunique() < 2:
    raise ValueError("Training data must contain both classes")
  X = df
  folds = StratifiedKFold(cv, shuffle=True, random_state=42)
  mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
  mlflow.set_experiment(experiment)
  best = None
  for name, (clf, grid) in MODELS.items():
    with mlflow.start_run(run_name=name):
      search = GridSearchCV(_pipeline(clone(clf)), grid, cv=folds, scoring="roc_auc").fit(X, y)
      scores = cross_validate(search.best_estimator_, X, y, cv=folds, scoring=SCORING)
      metrics = {k: float(scores[f"test_{k}"].mean()) for k in SCORING}
      mlflow.log_params({"model": name, "cv": cv, **search.best_params_})
      mlflow.log_metrics(metrics)
      mlflow.log_metrics({f"{k}_std": float(scores[f"test_{k}"].std()) for k in SCORING})
      _log_plots(search.best_estimator_, X, y, folds, name)
      mlflow.sklearn.log_model(search.best_estimator_, name="model",
                               serialization_format="cloudpickle")
      if best is None or metrics["roc_auc"] > best["metrics"]["roc_auc"]:
        best = {"best_model": name, "metrics": metrics, "estimator": search.best_estimator_}
      print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
  best["estimator"].fit(X, y)
  os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
  joblib.dump(best["estimator"], model_path)
  return {"best_model": best["best_model"], "metrics": best["metrics"]}


if __name__ == "__main__":
  print(train("data/processed/heart_clean.csv"))
