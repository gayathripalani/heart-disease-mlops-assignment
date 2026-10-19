#!/usr/bin/env bash
# One-shot pipeline for the lab (Linux/macOS/Git Bash): venv -> data -> EDA -> tests -> train.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
pip install -r requirements-dev.txt
export PYTHONPATH=src
python scripts/download_data.py
python scripts/eda.py
ruff check src tests scripts
pytest -q
python -m heart.train
echo "Done. Model: models/model.joblib. Next: mlflow ui --backend-store-uri sqlite:///mlflow.db"
