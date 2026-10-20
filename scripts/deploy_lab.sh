#!/usr/bin/env bash
# Lab deploy (Linux, no root): image -> local kind cluster -> k8s manifests -> smoke test.
# Needs docker or podman and internet access. Run after scripts/run_all.sh (needs models/model.joblib).
#   bash scripts/deploy_lab.sh              build, deploy, port-forward on :8080, smoke test
#   bash scripts/deploy_lab.sh monitoring   also start Prometheus + Grafana (needs docker compose)
#   bash scripts/deploy_lab.sh clean        delete the cluster and stop port-forward / monitoring
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER=heart
IMAGE=heart-api:1.0
PORT=8080
BIN="$HOME/bin"
mkdir -p "$BIN"
export PATH="$BIN:$PATH"

ENGINE=$(command -v docker || command -v podman || true)
[ -n "$ENGINE" ] || { echo "Neither docker nor podman found; cannot build or run a cluster here."; exit 1; }
ENGINE=$(basename "$ENGINE")
[ "$ENGINE" = podman ] && export KIND_EXPERIMENTAL_PROVIDER=podman
echo "Container engine: $ENGINE"

if [ "${1:-}" = clean ]; then
  [ -f .portforward.pid ] && kill "$(cat .portforward.pid)" 2>/dev/null || true
  rm -f .portforward.pid
  kind delete cluster --name "$CLUSTER" || true
  [ -f monitoring/docker-compose.lab.yml ] && $ENGINE compose -f monitoring/docker-compose.lab.yml down || true
  exit 0
fi

[ -f models/model.joblib ] || { echo "models/model.joblib missing; run: bash scripts/run_all.sh"; exit 1; }

if ! command -v kubectl >/dev/null; then
  echo "Installing kubectl to $BIN"
  curl -fsSL -o "$BIN/kubectl" "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x "$BIN/kubectl"
fi
if ! command -v kind >/dev/null; then
  echo "Installing kind to $BIN"
  curl -fsSL -o "$BIN/kind" https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
  chmod +x "$BIN/kind"
fi

echo "== Build image"
$ENGINE build -t "$IMAGE" .

echo "== Cluster"
kind get clusters 2>/dev/null | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER"
kubectl config use-context "kind-$CLUSTER" >/dev/null
$ENGINE save "$IMAGE" -o /tmp/heart-api.tar
kind load image-archive /tmp/heart-api.tar --name "$CLUSTER"

echo "== Deploy"
kubectl apply -f k8s/
kubectl rollout status deployment/heart-api --timeout=180s
kubectl get pods,svc

echo "== Port-forward :$PORT -> service heart-api:80"
[ -f .portforward.pid ] && kill "$(cat .portforward.pid)" 2>/dev/null || true
nohup kubectl port-forward --address 0.0.0.0 svc/heart-api "$PORT:80" >/tmp/portforward.log 2>&1 &
echo $! > .portforward.pid
for _ in $(seq 1 20); do curl -sf "localhost:$PORT/health" >/dev/null && break || sleep 1; done

echo "== Smoke test"
curl -sf "localhost:$PORT/health" && echo
curl -sf -X POST "localhost:$PORT/predict" -H "Content-Type: application/json" -d @docs/sample_request.json && echo

if [ "${1:-}" = monitoring ]; then
  echo "== Monitoring"
  sed "s|host.docker.internal:80|host.docker.internal:$PORT|" monitoring/prometheus.yml > monitoring/prometheus.lab.yml
  sed "s|./prometheus.yml|./prometheus.lab.yml|" monitoring/docker-compose.yml > monitoring/docker-compose.lab.yml
  $ENGINE compose -f monitoring/docker-compose.lab.yml up -d
  python scripts/send_traffic.py "http://localhost:$PORT" 200 || true
  echo "Prometheus http://localhost:9090/targets   Grafana http://localhost:3000"
fi

echo
echo "Done. API: http://localhost:$PORT/docs   Logs: kubectl logs deploy/heart-api"
echo "Note: kind has no cloud load balancer, so EXTERNAL-IP shows <pending>; the API is reached via port-forward."
