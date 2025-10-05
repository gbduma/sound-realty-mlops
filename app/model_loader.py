import json
import os
from pathlib import Path
import joblib
from typing import Tuple, List, Any

def load_model(model_dir: str = "model/") -> Tuple[Any, List[str], str]:
    model_path = Path(model_dir) / "model.pkl"
    feats_path = Path(model_dir) / "model_features.json"

    if not model_path.exists() or not feats_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Expected {model_path} and {feats_path}. "
            "Run `python create_model.py` first."
        )

    model = joblib.load(model_path)
    with open(feats_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    # Simple versioning: directory name or env override
    model_version = os.getenv("MODEL_VERSION", Path(model_dir).resolve().name or "local")
    return model, feature_order, model_version
