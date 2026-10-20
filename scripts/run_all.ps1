# One-shot pipeline for Windows labs (PowerShell): venv -> data -> EDA -> tests -> train.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$env:PYTHONPATH = "src"
python scripts/download_data.py
if ($LASTEXITCODE) { exit $LASTEXITCODE }
python scripts/eda.py
if ($LASTEXITCODE) { exit $LASTEXITCODE }
ruff check src tests scripts
if ($LASTEXITCODE) { exit $LASTEXITCODE }
pytest -q
if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m heart.train
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Write-Host "Done. Model: models/model.joblib. Next: python -m mlflow ui --backend-store-uri sqlite:///mlflow.db (with the venv active)"
