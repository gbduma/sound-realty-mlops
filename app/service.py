# app/service.py
import json
import time, os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .features import build_features, load_demographics
from .model_loader import load_model

NUMERIC_KINDS = {"i", "u", "f"}  # int, unsigned, float

def _coerce_df_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype.kind in NUMERIC_KINDS or df[c].dtype == "O":
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def _py(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    return v

class PredictionService:
    def __init__(self):
        # Load model & feature order
        self.model, self.feature_order, self.model_version = load_model(os.getenv("MODEL_URI", "model/"))

        # Load demographics (for backend join)
        self.demo_df = load_demographics(os.getenv("DEMO_URI", "data/zipcode_demographics.csv"))

        # Determine which model features come from demographics vs request
        demo_cols = set(self.demo_df.columns) - {"zipcode"}  # everything except the join key
        self.required_request_features: List[str] =["zipcode"] + [f for f in self.feature_order 
                                                                  if f not in demo_cols and f != "zipcode"]

        # Preload metrics (if present)
        self._metrics_path = Path(os.getenv("METRICS_PATH", "model/metrics.json"))
        self._metadata_cache = None
        if self._metrics_path.exists():
            try:
                self._metadata_cache = json.loads(self._metrics_path.read_text(encoding="utf-8"))
            except Exception:
                self._metadata_cache = None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns training-time metrics and info if available, with some live service info.
        """
        base = {
            "model_version": self.model_version,
            "feature_count": len(self.feature_order),
            "required_request_features": self.required_request_features,
        }
        if self._metadata_cache:
            base.update(self._metadata_cache)
        return base

    def _predict_from_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()

        X_in = pd.DataFrame(rows)
        X_in = _coerce_df_numeric(X_in)

        # Build engineered features and join demographics
        X = build_features(X_in, self.demo_df)

        # Ensure we have all model features (will KeyError if missing)
        X = X[self.feature_order]

        # Snapshot model inputs for response
        features_snapshot = {k: _py(X.iloc[0][k]) for k in self.feature_order}

        # Predict
        yhat = float(self.model.predict(X)[0])
        return {
            "prediction": yhat,
            "model_version": self.model_version,
            "features_used": features_snapshot,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    def predict_full(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Full payload should mirror the future_unseen_examples.csv columns (no demographics)
        return self._predict_from_rows([row])

    def predict_minimal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept only the strictly required request-side fields.
        We join demographics and build the final matrix internally.
        """
        missing = [k for k in self.required_request_features if k not in payload]
        if missing:
            raise ValueError(f"Missing required fields for minimal endpoint: {missing}")

        # Only pass the minimal set forward; extra keys are ignored to keep contract tight
        row = {k: payload[k] for k in self.required_request_features}
        return self._predict_from_rows([row])
