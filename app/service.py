import json, time, os
import numpy as np
import pandas as pd
from .features import build_features, load_demographics
from .model_loader import load_model

def _py(v):
    # turn numpy types into plain python scalars for JSON
    if isinstance(v, (np.generic,)):
        return v.item()
    return v

class PredictionService:
    def __init__(self):
        self.model, self.feature_order, self.model_version = load_model(os.getenv("MODEL_URI", "model/"))
        self.demo_df = load_demographics(os.getenv("DEMO_URI", "data/zipcode_demographics.csv"))

    def predict_full(self, row: Dict[str, Any]):
        t0 = time.time()
        X_in = pd.DataFrame([row])
        X_in = _coerce_df_numeric(X_in)

        # Build engineered features and join demographics
        X = build_features(X_in, self.demo_df)
        # Reorder to the model's expected feature order
        X = X[self.feature_order]

        # Snapshot model inputs for the response
        features_snapshot = {k: _py(X.iloc[0][k]) for k in self.feature_order}

        # Predict
        yhat = float(self.model.predict(X)[0])

        return {
            "prediction": yhat,
            "model_version": self.model_version,
            "features_used": features_snapshot,  # <-- now shows engineered + joined values
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    def predict_minimal(self, payload: dict):
        # validate required keys exist according to feature_order; build/engineer same as full
        return self.predict_full(payload)
