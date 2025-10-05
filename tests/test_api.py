from pathlib import Path
import subprocess
import sys
import pandas as pd
from fastapi.testclient import TestClient

# Import after potential model build to avoid startup failures
def ensure_model():
    model_pkl = Path("model/model.pkl")
    feats_json = Path("model/model_features.json")
    if not (model_pkl.exists() and feats_json.exists()):
        subprocess.check_call([sys.executable, "create_model.py"])

def test_health_and_predict_full():
    ensure_model()

    # Lazy import so app can find model artifacts at startup
    from app.main import app

    client = TestClient(app)

    # Health check
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_version" in r.json()

    # Use the first future example
    df = pd.read_csv("data/future_unseen_examples.csv")
    assert len(df) > 0, "future_unseen_examples.csv is empty"

    payload = df.iloc[0].to_dict()
    # Make sure we don't accidentally send prohibited fields (usually not present here, but just in case)
    for bad in ("price", "date", "id"):
        payload.pop(bad, None)

    r = client.post("/predict/full", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
