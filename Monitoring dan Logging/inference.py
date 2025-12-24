import os
import json
import time
import requests
import pandas as pd

MLFLOW_URL = "http://127.0.0.1:5000"
EXPORTER_URL = "http://127.0.0.1:8000"
MODEL_NAME = "dicoding-model"
MODEL_VERSION = "v1"

INVOCATIONS_URL = f"{MLFLOW_URL}/invocations"


def load_preprocessed_data(csv_path, n=5):
    df = pd.read_csv(csv_path)
    df = df.head(n)

    # pastikan numeric (aman)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    return df


def to_mlflow_payload(df):
    return {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": df.values.tolist()
        }
    }


def send_inference(payload):
    payload_json = json.dumps(payload)
    payload_bytes = len(payload_json.encode("utf-8"))

    status_code = 0
    response_bytes = 0
    latency = 0.0
    error_type = None

    t0 = time.time()
    try:
        r = requests.post(
            INVOCATIONS_URL,
            headers={"Content-Type": "application/json"},
            data=payload_json,
            timeout=20
        )
        latency = time.time() - t0
        status_code = r.status_code
        response_bytes = len(r.content or b"")

        if status_code != 200:
            error_type = f"http_{status_code}"

    except Exception as e:
        latency = time.time() - t0
        error_type = type(e).__name__

    track_payload = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "status_code": status_code,
        "latency_seconds": latency,
        "payload_bytes": payload_bytes,
        "response_bytes": response_bytes,
        "error_type": error_type
    }

    # kirim metrik ke exporter
    try:
        requests.post(
            f"{EXPORTER_URL}/track_inference",
            json=track_payload,
            timeout=3
        )
    except Exception:
        pass

    return track_payload


def main():
    # BACA env DI SINI (bukan di global)
    data_csv = os.getenv("DATA_CSV", "").strip()
    print("DATA_CSV terbaca:", repr(data_csv))

    if not data_csv or not os.path.exists(data_csv):
        raise RuntimeError(f"❌ DATA_CSV belum valid / file tidak ditemukan: {data_csv}")

    df = load_preprocessed_data(data_csv, n=5)
    payload = to_mlflow_payload(df)

    print("🚀 Inference berjalan (pakai X_test.csv)...")
    while True:
        result = send_inference(payload)
        print(result)
        time.sleep(1)


if __name__ == "__main__":
    main()
