# app/service.py
import json
import time, os
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .features import build_features, load_demographics
from .model_loader import load_model_from_dir
from concurrent.futures import ThreadPoolExecutor
from .comps import CompsEstimator
from .quality import OODGuard, basic_rules
from .s3_registry import sync_model_from_s3

class PredictionService:
    def __init__(self):
        # Load model & feature order
        # registry dir and default model
        self.registry_dir = os.getenv("MODEL_REGISTRY_DIR", "model")
        self.default_model_name = os.getenv("MODEL_DEFAULT", "enhanced_auto")
        self.models = {}  # name -> (model, feature_order, version)

        # If MODEL_REGISTRY_S3_PREFIX set, ensure default model exists locally
        s3_prefix = os.getenv("MODEL_REGISTRY_S3_PREFIX")
        if s3_prefix:
            sync_model_from_s3(s3_prefix, self.default_model_name, self.registry_dir)
            # optionally ensure the baseline is present too:
            try:
                sync_model_from_s3(s3_prefix, "baseline_knn", self.registry_dir)
            except Exception:
                pass
        
        # preload default
        self._ensure_loaded(self.default_model_name)

        # Load demographics (for backend join)
        self.demo_df = load_demographics(os.getenv("DEMO_URI", "data/zipcode_demographics.csv"))

        # Build training frame for comps & OOD (load once)
        self.sales_df = pd.read_csv("data/kc_house_data.csv")
        self.comps = CompsEstimator(self.sales_df)
        # Fit OOD on the SAME features model sees after build_features()
        tmp = self.sales_df[[c for c in self.sales_df.columns if c != "price"]].copy()
        tmp["zipcode"] = tmp["zipcode"].astype(str)
        tmp = build_features(tmp, self.demo_df)
        self.ood = OODGuard(tmp)
        self.pool = ThreadPoolExecutor(max_workers=4)

        # optional metrics hot-reload bookkeeping
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._metrics_mtime: Dict[str, float] = {}
        
    def _ensure_loaded(self, name: str):
        """Load model artifacts from model/<name>/ if not already cached."""
        if name in self.models:
            return
        model_dir = os.path.join(self.registry_dir, name)
        self.models[name] = load_model_from_dir(model_dir)

    def _choose_model(self, name: Optional[str]):
        """Return ((model, feature_order, version), name) using default when None."""
        name = name or self.default_model_name
        self._ensure_loaded(name)
        return self.models[name], name
    
    def _try_load_metrics(self, model_name: str) -> Dict[str, Any]:
        """Hot-reload metrics.json under model/<model_name>/ if mtime changed."""
        p = Path(self.registry_dir) / model_name / "metrics.json"
        if not p.exists():
            return {}
        mtime = p.stat().st_mtime
        if model_name not in self._metrics_cache or self._metrics_mtime.get(model_name) != mtime:
            try:
                self._metrics_cache[model_name] = json.loads(p.read_text(encoding="utf-8"))
                self._metrics_mtime[model_name] = mtime
            except Exception:
                self._metrics_cache[model_name] = {}
        return self._metrics_cache.get(model_name, {})

    def get_metadata_for(self, model_name: Optional[str]):
        """Assemble metadata for the selected model."""
        (model_obj, feat_order, version) = self.models[model_name]
        meta = {
            "model_version": version,
            "feature_count": len(feat_order),
        }
        meta.update(self._try_load_metrics(model_name))

        # required request features = zipcode + non-demographic model features
        demo_cols = set(self.demo_df.columns) - {"zipcode"}
        required = ["zipcode"] + [f for f in feat_order if f not in demo_cols and f != "zipcode"]
        meta["required_request_features"] = required
        return meta

    def _predict_core(self, row: dict, model_name: Optional[str]):
        model, feature_order, version = self.models[model_name]
        t0 = time.time()
        # kick off side tasks
        fut_comps = self.pool.submit(self.comps.estimate, row)
        fut_rules = self.pool.submit(basic_rules, row)

        # core feature build + predict
        # Build engineered features and join demographics
        X = build_features(pd.DataFrame([row]), self.demo_df)

        # Ensure we have all model features (will KeyError if missing)
        X = X[feature_order]

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
            "model_version": version,
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
