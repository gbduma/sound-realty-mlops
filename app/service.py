import json, time, os
import pandas as pd
from .features import build_features, load_demographics
from .model_loader import load_model

class PredictionService:
    def __init__(self):
        self.model, self.feature_order, self.model_version = load_model(os.getenv("MODEL_URI", "model/"))
        self.demo_df = load_demographics(os.getenv("DEMO_URI", "data/zipcode_demographics.csv"))

    def predict_full(self, row: dict):
        t0 = time.time()
        X = build_features(pd.DataFrame([row]), self.demo_df)
        X = X[self.feature_order]
        yhat = float(self.model.predict(X)[0])
        return {
            "prediction": yhat,
            "model_version": self.model_version,
            "features_used": {k: (row.get(k, None)) for k in self.feature_order},
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    def predict_minimal(self, payload: dict):
        # validate required keys exist according to feature_order; build/engineer same as full
        return self.predict_full(payload)
