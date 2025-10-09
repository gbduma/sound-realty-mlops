from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, Type, List
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

class CompsInfo(BaseModel):
    comp_count: int = Field(0, description="Number of comps used")
    comps_estimate: Optional[float] = Field(None, description="Median price of comps")
    comps_band: Optional[Tuple[float, float]] = Field(None, description="(p5, p95) band for comps")
    avg_dist_km: Optional[float] = Field(None, description="Mean distance (km) of comps")

class QualityInfo(BaseModel):
    ood_score: Optional[float] = Field(None, description="Out-of-distribution score (0–1, higher is more typical)")
    confidence: Optional[float] = Field(None, description="Heuristic confidence (0–1)")
    messages: List[str] = Field(default_factory=list, description="Validation messages")

class AuxOut(BaseModel):
    comps: Optional[CompsInfo] = None
    quality: Optional[QualityInfo] = None

class PredictionOut(BaseModel):
    prediction: float
    model_version: str
    features_used: Dict[str, Any]
    aux: Optional[AuxOut] = None
    latency_ms: float

