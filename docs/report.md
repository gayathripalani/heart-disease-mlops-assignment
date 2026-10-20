# Heart Disease Risk Prediction: MLOps Report

Name / Roll no: **Gayathri Palani** / **2025AE05661** (2025ae05661@wilp.bits-pilani.ac.in)  |  Repo: **https://github.com/gayathripalani/heart-disease-mlops-assignment**  |  API: `http://localhost:8080/predict` (local Kubernetes via port-forward)

## 1. Setup
See `README.md`. Python 3.12, `pip install -r requirements-dev.txt`, then `bash scripts/run_all.sh`.
Data comes from https://archive.ics.uci.edu/dataset/45/heart+disease (Cleveland file, 303 rows).

## 2. Data and EDA
- 303 patients, 13 features + binary target (`num > 0` = disease). Class balance 54% / 46%.
- Missing values: `ca` (4) and `thal` (2) only; imputed with median / most frequent.
- Figures: `docs/screenshots/eda/` (class balance, histograms, correlation heatmap, boxplots, categorical counts).
- **Add your own 3-4 observations from the plots here** (for example which features separate the classes most).

## 3. Features and models
- Numeric: median impute + standard scale. Categorical: mode impute + one-hot (unknown categories ignored).
- Engineered: `hr_reserve = (220 - age) - thalach`, `chol_age_ratio = chol / age`.
- Models tuned with `GridSearchCV` (5-fold stratified, ROC-AUC) then re-scored with 5-fold cross-validation:

| Model | Best params | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | C=0.1 | 0.868 | 0.896 | 0.805 | 0.919 |
| Random Forest | n_estimators=100, max_depth=5 | 0.848 | 0.873 | 0.784 | 0.918 |
| Gradient Boosting | n_estimators=100, lr=0.05 | 0.805 | 0.819 | 0.748 | 0.888 |

Selected: Logistic Regression (best ROC-AUC, simplest, well calibrated for a small dataset). Recall matters
most for screening, so false negatives are the main error to watch. Note the tuning folds and scoring folds
share data, so scores are slightly optimistic.

## 4. Experiment tracking
MLflow (SQLite backend) logs parameters, mean and std of every metric, ROC curve and confusion matrix per run,
and the fitted pipeline. View with `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db` (venv active).
**Screenshot: `docs/screenshots/mlflow_runs.png`**

## 5. Packaging and reproducibility
One scikit-learn `Pipeline` (feature engineering + preprocessing + classifier) saved with joblib, so the API
sends raw fields only. Pinned `requirements.txt`, fixed random seeds.

## 6. CI/CD
`.github/workflows/ci.yml`: lint, tests with coverage, training, artifacts, Docker build and smoke test.
**Screenshots: green run, red run (deliberate failure), artifacts list.**

## 7. Containerisation
`Dockerfile` (python:3.12-slim, non-root, health check). **Screenshots: build, run, curl response.**

## 8. Kubernetes deployment
`k8s/deployment.yaml` (1 replica so Prometheus sees a single series; probes, resource limits) and `k8s/service.yaml` (LoadBalancer).
**Screenshots: `kubectl get pods,svc`, curl through the service.**

## 9. Monitoring and logging
JSON access logs (`kubectl logs`), Prometheus metrics (`predict_requests_total`, `predict_latency_seconds`,
`predict_positive_total`), Grafana dashboard `monitoring/grafana/dashboards/heart-api.json`.
**Screenshots: Prometheus targets, Grafana dashboard under load.**

## 10. Architecture
Render `docs/architecture.mmd` with `npx @mermaid-js/mermaid-cli -i docs/architecture.mmd -o docs/architecture.png`.

## 11. Limitations
Small single-site dataset, no external validation, not a clinical tool.
