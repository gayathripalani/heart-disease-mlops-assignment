"""Send sample /predict requests so the Grafana dashboard has data.

Usage: python scripts/send_traffic.py [base_url] [count]
"""
import json
import random
import sys
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
sample = json.load(open("docs/sample_request.json"))
ok = 0
for _ in range(count):
  body = {**sample, "age": random.randint(30, 75), "chol": random.randint(150, 350),
          "thalach": random.randint(90, 200), "oldpeak": round(random.random() * 4, 1)}
  req = urllib.request.Request(f"{base}/predict", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
  with urllib.request.urlopen(req) as r:
    ok += r.status == 200
print(f"{ok}/{count} requests succeeded")
