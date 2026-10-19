# Heart Disease Risk: End-to-End MLOps

Classifier for heart-disease risk (UCI Heart Disease, Cleveland subset) served as a FastAPI service,
tracked with MLflow, tested and built in GitHub Actions, packaged in Docker, deployed to Kubernetes
and monitored with Prometheus + Grafana.

Repo: https://github.com/gayathripalani/heart-disease-mlops

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

## Quick start (lab machine)

Needs Python 3.12. One command runs everything up to a trained model:

```bash
bash scripts/run_all.sh          # Git Bash / Linux / macOS
powershell -File scripts/run_all.ps1   # Windows PowerShell
```

Or step by step:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export PYTHONPATH=src                                  # PowerShell: $env:PYTHONPATH="src"
python scripts/download_data.py      # downloads UCI zip, writes data/processed/heart_clean.csv
python scripts/eda.py                # figures in docs/screenshots/eda/
pytest -q                            # unit tests
python -m heart.train                # trains 3 models, logs to MLflow, saves models/model.joblib
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

## Run the API locally

```bash
PYTHONPATH=src uvicorn api.app:create_app --factory --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @docs/sample_request.json
# {"prediction":0,"confidence":0.63}
```

Endpoints: `POST /predict`, `GET /health`, `GET /metrics`, interactive docs at `/docs`.
`ca` and `thal` are optional (imputed); other fields are range-checked (422 on bad input).

## Docker

```bash
python -m heart.train                       # model must exist in models/
docker build -t heart-api:1.0 .
docker run --rm -p 8000:8000 heart-api:1.0
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @docs/sample_request.json
```

## Kubernetes (Docker Desktop with Kubernetes enabled, or Minikube)

```bash
kubectl apply -f k8s/
kubectl get pods,svc                        # EXTERNAL-IP localhost on Docker Desktop
curl -X POST http://localhost/predict -H "Content-Type: application/json" -d @docs/sample_request.json
```
Minikube: `minikube image load heart-api:1.0`, then `minikube service heart-api --url`
(update the Prometheus target accordingly).

## Monitoring

```bash
cd monitoring && docker compose up -d       # Prometheus :9090, Grafana :3000 (dashboard "Heart API")
python scripts/send_traffic.py http://localhost 200
kubectl logs deploy/heart-api               # JSON request logs
```

## CI/CD

`.github/workflows/ci.yml`: ruff lint -> pytest (+coverage) -> model training -> upload reports and
model as artifacts -> Docker build and container smoke test. Any lint or test failure fails the run.

## Layout

```
src/heart/    data, features, train, predict      src/api/   FastAPI app + schemas
scripts/      download, EDA, run_all, traffic     tests/     pytest suite
k8s/          Deployment + Service                monitoring/ Prometheus + Grafana
docs/         report, screenshots, sample request notebooks/ EDA notebook
```
