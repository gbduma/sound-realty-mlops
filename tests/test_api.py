# tests/test_api.py
from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import pandas as pd
from fastapi.testclient import TestClient


def ensure_model():
    model_pkl = Path("model/model.pkl")
    feats_json = Path("model/model_features.json")
    if not (model_pkl.exists() and feats_json.exists()):
        subprocess.check_call([sys.executable, "create_model.py"])


def test_health_and_predict_full():
    ensure_model()

    # Import app after model is ensured to avoid startup errors
    from app.main import app  # noqa: WPS433

    client = TestClient(app)

    # health
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_version" in r.json()

    # pick a real row
    df = pd.read_csv("data/future_unseen_examples.csv")
    assert not df.empty, "future_unseen_examples.csv is empty"
    payload = df.iloc[0].to_dict()
    for bad in ("price", "date", "id"):
        payload.pop(bad, None)

    r = client.post("/predict/full", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
