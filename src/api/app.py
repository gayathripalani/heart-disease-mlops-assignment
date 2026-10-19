import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import PatientFeatures, Prediction
from heart.predict import load_model, predict_one

log = logging.getLogger("heart-api")
if not log.handlers:
  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter("%(message)s"))
  log.addHandler(handler)
  log.setLevel(logging.INFO)

REQUESTS = Counter("predict_requests_total", "Predict calls", ["outcome"])
for _outcome in ("ok", "error"):
  REQUESTS.labels(_outcome)
LATENCY = Histogram("predict_latency_seconds", "Predict latency")
POSITIVES = Counter("predict_positive_total", "Positive predictions")


def create_app() -> FastAPI:
  @asynccontextmanager
  async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield

  app = FastAPI(title="Heart Disease Risk API", lifespan=lifespan)

  @app.middleware("http")
  async def access_log(request: Request, call_next):
    start = time.perf_counter()
    resp = await call_next(request)
    log.info(json.dumps({"path": request.url.path, "method": request.method,
                         "status": resp.status_code,
                         "ms": round((time.perf_counter() - start) * 1000, 2)}))
    return resp

  @app.post("/predict", response_model=Prediction)
  def predict(body: PatientFeatures, request: Request):
    with LATENCY.time():
      try:
        out = predict_one(request.app.state.model, body.model_dump())
      except Exception:
        REQUESTS.labels("error").inc()
        log.exception("prediction failed")
        raise HTTPException(500, "prediction failed")
    REQUESTS.labels("ok").inc()
    POSITIVES.inc(out["prediction"])
    log.info(json.dumps({"event": "prediction", **out}))
    return out

  @app.get("/health")
  def health(request: Request):
    if getattr(request.app.state, "model", None) is None:
      raise HTTPException(503, "model not loaded")
    return {"status": "ok"}

  @app.get("/metrics")
  def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

  return app
