# app/service.py
import json
import time, os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .features import build_features, load_demographics
from .model_loader import load_model_from_dir
from concurrent.futures import ThreadPoolExecutor
from .comps import CompsEstimator
from .quality import OODGuard, basic_rules

NUMERIC_KINDS = {"i", "u", "f"}  # int, unsigned, float

def _coerce_df_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype.kind in NUMERIC_KINDS or df[c].dtype == "O":
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

class PredictionService:
    def __init__(self):
        # Load model & feature order
        # registry dir and default model
        self.registry_dir = os.getenv("MODEL_REGISTRY_DIR", "model")
        self.default_model_name = os.getenv("MODEL_DEFAULT", "enhanced_auto")
        self.models = {}  # name -> (model, feature_order, version)
        # preload default
        self._ensure_loaded(self.default_model_name)

        # Load demographics (for backend join)
        self.demo_df = load_demographics(os.getenv("DEMO_URI", "data/zipcode_demographics.csv"))

        # Determine which model features come from demographics vs request
        demo_cols = set(self.demo_df.columns) - {"zipcode"}  # everything except the join key
        self.required_request_features: List[str] =["zipcode"] + [f for f in self.feature_order 
                                                                  if f not in demo_cols and f != "zipcode"]
        # Build training frame for comps & OOD (load once)
        self.sales_df = pd.read_csv("data/kc_house_data.csv")
        self.comps = CompsEstimator(self.sales_df)
        # Fit OOD on the SAME features model sees after build_features()
        tmp = self.sales_df[[c for c in self.sales_df.columns if c != "price"]].copy()
        tmp["zipcode"] = tmp["zipcode"].astype(str)
        tmp = build_features(tmp, self.demo_df)
        self.ood = OODGuard(tmp)
        self.pool = ThreadPoolExecutor(max_workers=4)

        # Preload metrics (if present)
        self._metrics_path = Path(os.getenv("METRICS_PATH", "model/metrics.json"))
        self._metadata_cache = None
        if self._metrics_path.exists():
            try:
                self._metadata_cache = json.loads(self._metrics_path.read_text(encoding="utf-8"))
            except Exception:
                self._metadata_cache = None
        
        def _ensure_loaded(self, name: str):
            if name in self.models:
                return
            model_dir = os.path.join(self.registry_dir, name)
            self.models[name] = load_model_from_dir(model_dir)

        def _choose_model(self, name: str | None):
            name = name or self.default_model_name
            self._ensure_loaded(name)
            return self.models[name], name

    def get_metadata_for(self, model_name: str):
        """
        Returns training-time metrics and info if available, with some live service info.
        """
        # look up metrics.json under model/model_name
        import json
        from pathlib import Path
        p = Path(self.registry_dir) / model_name / "metrics.json"
        base = {
            "model_version":  model_name,
            "feature_count": len(self.models[model_name][1]),
            "required_request_features": self.required_request_features,
        }
        if p.exists():
            try:
                base.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        # Also add minimal required fields, including zipcode
        demo_cols = set(self.demo_df.columns) - {"zipcode"}
        feat_order = self.models[model_name][1]
        req = ["zipcode"] + [f for f in feat_order if f not in demo_cols and f != "zipcode"]
        base["required_request_features"] = req
        return base

    def _predict_core(self, row: dict, model_name: str):
        model, feature_order, version = self.models[model_name]
        # build features (engineered features were baked during training; here we just align)
        import pandas as pd, time
        from .features import build_features  # we can call with add_engineered=True; it won't hurt
        t0 = time.time()
        # kick off side tasks
        fut_comps = self.pool.submit(self.comps.estimate, row)
        fut_rules = self.pool.submit(basic_rules, row)

        # core feature build + predict
        # Build engineered features and join demographics
        X = build_features(pd.DataFrame([row]), self.demo_df)

        # Ensure we have all model features (will KeyError if missing)
        X = X[self.feature_order]

        # Snapshot model inputs for response
        features_snapshot = {k: (float(X.iloc[0][k]) if hasattr(X.iloc[0][k], "item") else X.iloc[0][k]) for k in feature_order}

        # Predict
        yhat = float(model.predict(X)[0])

        # collect side results
        comps_info = fut_comps.result(timeout=1.5)
        rules_info = fut_rules.result(timeout=0.1)
        ood_score = self.ood.score(features_snapshot)

        # confidence heuristic
        conf = 1.0
        if comps_info.get("comp_count", 0) < 5: conf -= 0.2
        if "comps_band" in comps_info:
            span = comps_info["comps_band"][1] - comps_info["comps_band"][0]
            if span > 500_000: conf -= 0.1
        conf -= rules_info.get("penalty", 0.0)
        conf *= 0.8 + 0.2*ood_score
        conf = float(max(0.0, min(1.0, conf)))

        return {
            "prediction": yhat,
            "model_version": self.model_version,
            "features_used": features_snapshot,
            "aux": {
                "comps": comps_info,
                "quality": {
                    "ood_score": ood_score,
                    "confidence": conf,
                    "messages": rules_info.get("messages", []),
                }
            },
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    def predict_full_with_model(self, row: dict, model_name: str):
        # Full payload should mirror the future_unseen_examples.csv columns (no demographics)
        return self._predict_core(row, model_name)

    def predict_minimal_with_model(self, payload: dict, model_name: str):
        """
        Accept only the strictly required request-side fields.
        We join demographics and build the final matrix internally.
        """
        demo_cols = set(self.demo_df.columns) - {"zipcode"}
        feat_order = self.models[model_name][1]
        required = ["zipcode"] + [f for f in feat_order if f not in demo_cols and f != "zipcode"]
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"Missing required fields for minimal endpoint: {missing}")
        row = {k: payload[k] for k in required}
        return self._predict_core(row, model_name)
