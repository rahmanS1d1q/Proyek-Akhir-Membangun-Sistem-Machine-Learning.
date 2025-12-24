import os
import time
import json
import threading
from typing import Optional, Dict, Any

import requests
from flask import Flask, request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# =========================
# Config
# =========================
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://127.0.0.1:5000")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8000"))
PROBE_INTERVAL_SEC = int(os.getenv("PROBE_INTERVAL_SEC", "5"))

# Optional label info for dashboard clarity
MODEL_NAME = os.getenv("MODEL_NAME", "dicoding-model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
RUN_ID = os.getenv("RUN_ID", "unknown")

app = Flask(__name__)

# =========================
# 3 METRIK "MLFLOW" (sesuai permintaan: matrix mlflow 3)
# =========================

# (1) apakah MLflow server up
mlflow_server_up = Gauge(
    "mlflow_server_up",
    "1 jika MLflow server dapat diakses, 0 jika tidak",
    ["mlflow_url"],
)

# (2) info model (gauge bernilai 1, label berisi identitas)
mlflow_model_info = Gauge(
    "mlflow_model_info",
    "Info identitas model yang diserve (value=1)",
    ["model_name", "model_version", "run_id"],
)

# (3) waktu respon endpoint health/probe MLflow (latency probe)
mlflow_probe_latency_seconds = Gauge(
    "mlflow_probe_latency_seconds",
    "Latency probe sederhana ke MLflow server",
    ["mlflow_url", "endpoint"],
)

# set model info sekali
mlflow_model_info.labels(
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
    run_id=RUN_ID,
).set(1)

# =========================
# 7 METRIK PERFORMA (matrix performance 7)
# =========================

# 1) total request inference yang tercatat
inference_requests_total = Counter(
    "inference_requests_total",
    "Total request inference yang dikirim ke MLflow server",
    ["model_name", "model_version"],
)

# 2) total error request inference
inference_request_errors_total = Counter(
    "inference_request_errors_total",
    "Total error request inference (HTTP != 200 atau exception)",
    ["model_name", "model_version", "error_type"],
)

# 3) latency inference (histogram)
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Latency request inference ke MLflow server",
    ["model_name", "model_version", "status_code"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

# 4) payload size (bytes)
inference_payload_bytes = Histogram(
    "inference_payload_bytes",
    "Ukuran payload request inference (bytes)",
    ["model_name", "model_version"],
    buckets=(100, 300, 1_000, 3_000, 10_000, 30_000, 100_000),
)

# 5) response size (bytes)
inference_response_bytes = Histogram(
    "inference_response_bytes",
    "Ukuran response inference (bytes)",
    ["model_name", "model_version"],
    buckets=(100, 300, 1_000, 3_000, 10_000, 30_000, 100_000),
)

# 6) request yang sedang berjalan
inference_in_progress = Gauge(
    "inference_in_progress",
    "Jumlah request inference yang sedang diproses",
    ["model_name", "model_version"],
)

# 7) timestamp request terakhir (epoch seconds)
inference_last_request_timestamp = Gauge(
    "inference_last_request_timestamp",
    "Waktu (epoch seconds) request inference terakhir",
    ["model_name", "model_version"],
)

# =========================
# Helper
# =========================

def _now() -> float:
    return time.time()

def _probe_mlflow() -> None:
    """Thread: periodically probe MLflow server."""
    while True:
        ok = 0
        try:
            # MLflow model serving biasanya punya endpoint root yang merespons 200/404 cepat.
            # Kita coba GET "/" untuk probe basic liveness.
            t0 = _now()
            r = requests.get(f"{MLFLOW_URL}/", timeout=2)
            dt = _now() - t0
            mlflow_probe_latency_seconds.labels(mlflow_url=MLFLOW_URL, endpoint="/").set(dt)

            # Anggap up jika dapat response apapun (status code valid)
            ok = 1
        except Exception:
            ok = 0

        mlflow_server_up.labels(mlflow_url=MLFLOW_URL).set(ok)
        time.sleep(PROBE_INTERVAL_SEC)

probe_thread = threading.Thread(target=_probe_mlflow, daemon=True)
probe_thread.start()

# =========================
# Routes
# =========================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.post("/track_inference")
def track_inference():
    """
    Endpoint untuk mencatat metrik performa inference.
    inference.py akan:
    - kirim payload JSON ke MLflow /invocations
    - lalu kirim hasil status/latency/bytes ke endpoint ini
    """
    body: Dict[str, Any] = request.get_json(force=True, silent=False)

    model_name = body.get("model_name", MODEL_NAME)
    model_version = body.get("model_version", MODEL_VERSION)

    status_code = str(body.get("status_code", "0"))
    latency = float(body.get("latency_seconds", 0.0))
    payload_bytes = int(body.get("payload_bytes", 0))
    response_bytes = int(body.get("response_bytes", 0))
    error_type = body.get("error_type")

    inference_requests_total.labels(model_name=model_name, model_version=model_version).inc()
    inference_last_request_timestamp.labels(model_name=model_name, model_version=model_version).set(_now())
    inference_payload_bytes.labels(model_name=model_name, model_version=model_version).observe(payload_bytes)
    inference_response_bytes.labels(model_name=model_name, model_version=model_version).observe(response_bytes)
    inference_latency_seconds.labels(
        model_name=model_name,
        model_version=model_version,
        status_code=status_code
    ).observe(latency)

    if error_type:
        inference_request_errors_total.labels(
            model_name=model_name,
            model_version=model_version,
            error_type=str(error_type),
        ).inc()

    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "mlflow_url": MLFLOW_URL}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=EXPORTER_PORT)
