# app/model_loader.py
from __future__ import annotations

import json, pickle
import os
from pathlib import Path
from typing import Any, List, Tuple

def load_model_from_dir(model_dir: str) -> Tuple[Any, List[str], str]:
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
    
    model = pickle.load(open(model_path, "rb"))
    feats = json.load(open(feats_path, "r", encoding="utf-8"))
    version = base.name  # e.g., "baseline_knn"/"enhanced_auto"
    return model, feats, version

