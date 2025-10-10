# tests/test_api.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

def test_health_and_predict_full():
    # Import AFTER artifacts are present (CI workflow trains first)
    from app.main import app  # noqa: WPS433

    client = TestClient(app)

    # ---- health: new shape (registry-aware) ----
    r = client.get("/health")
    assert r.status_code == 200
    health = r.json()
    # expected keys
    for k in ("status", "default_model", "selected_model", "loaded_models"):
        assert k in health, f"missing '{k}' in /health response"
    assert health["status"] == "ok"
    # on cold start, selected_model should equal default_model
    assert health["selected_model"] == health["default_model"]

    # ---- metadata: both models should resolve (baseline + enhanced) ----
    for name in ("baseline_knn", "enhanced_auto"):
        r = client.get(f"/model/metadata?model={name}")
        assert r.status_code == 200, f"/model/metadata failed for {name}: {r.text}"
        meta = r.json()
        assert meta.get("selected_model") == name
        assert meta.get("feature_count", 0) > 0

    # ---- pick a real row from future_unseen_examples ----
    df = pd.read_csv("data/future_unseen_examples.csv")
    assert not df.empty, "future_unseen_examples.csv is empty"
    payload = df.iloc[0].to_dict()
    # remove only columns that are not part of requests
    for bad in ("price", "date", "id"):
        payload.pop(bad, None)

    # ---- predict/full with default (enhanced_auto) ----
    r = client.post("/predict/full", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "prediction" in body and isinstance(body["prediction"], (int, float))
    assert body.get("model_version") in ("baseline_knn", "enhanced_auto")

    # ---- predict/full explicitly with baseline_knn ----
    r = client.post("/predict/full?model=baseline_knn", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("model_version") == "baseline_knn"
    assert isinstance(body["prediction"], (int, float))

    # ---- predict/full explicitly with enhanced_auto ----
    r = client.post("/predict/full?model=enhanced_auto", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("model_version") == "enhanced_auto"
    assert isinstance(body["prediction"], (int, float))