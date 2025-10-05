from typing import Optional, Dict, Any, Tuple, Type
import pandas as pd
from pydantic import BaseModel, create_model, Field

EXCLUDE_COLS = {"price", "date", "id"}  # not expected from clients

def _pyd_type_for_series(s: pd.Series) -> Type:
    s = s.dropna()
    if s.empty:
        # unknown -> accept string; we'll coerce later
        return Optional[str]
    if pd.api.types.is_integer_dtype(s):
        return Optional[int]
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return Optional[float]
    if pd.api.types.is_bool_dtype(s):
        return Optional[bool]
    return Optional[str]

def build_full_input_model(csv_path: str = "data/future_unseen_examples.csv") -> Type[BaseModel]:
    df = pd.read_csv(csv_path, nrows=1000)  # sample for dtype inference
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    fields: Dict[str, Tuple[Type, Any]] = {}
    for c in cols:
        fields[c] = (_pyd_type_for_series(df[c]), Field(..., description=f"Auto-inferred from {csv_path}"))
    # Dynamically create Pydantic model with required fields matching CSV
    Model = create_model("FullInput", **fields)  # type: ignore
    return Model

# Build the concrete model at import time
FullInput = build_full_input_model()

class MinimalInput(BaseModel):
    # accepts a subset of required features (we’ll validate against model_features.json in service.py)
    payload: Dict[str, Any]

class PredictionOut(BaseModel):
    prediction: float
    model_version: str
    features_used: Dict[str, Any]
    latency_ms: float
