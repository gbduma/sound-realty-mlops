# app/model_loader.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Tuple

import joblib


def load_model(model_dir: str = "model/") -> Tuple[Any, List[str], str]:
    """
    Load the trained model and its feature order from disk.

    Returns:
        (model, feature_order, model_version)
    """
    base = Path(model_dir)
    model_path = base / "model.pkl"
    feats_path = base / "model_features.json"

    if not model_path.exists() or not feats_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Expected {model_path} and {feats_path}. "
            "Run `python create_model.py` first."
        )

    model = joblib.load(model_path)
    with feats_path.open("r", encoding="utf-8") as f:
        feature_order = json.load(f)

    # Allow overriding via env; otherwise use folder name or 'local'
    model_version = os.getenv("MODEL_VERSION") or (base.resolve().name or "local")
    return model, feature_order, model_version
