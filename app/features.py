# app/features.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

# choose a sensible reference year; KC dataset centers around 2014–2015
CURRENT_YEAR = 2015

def _series_default(df: pd.DataFrame, value) -> pd.Series:
    """Return a Series of length len(df) filled with `value`."""
    return pd.Series(value, index=df.index)

def _num_col(df: pd.DataFrame, name: str, default=0) -> pd.Series:
    """
    Safely fetch a numeric column as a Series aligned to df.index.
    If missing, return a Series filled with `default`.
    """
    base = df[name] if name in df.columns else _series_default(df, default)
    return pd.to_numeric(base, errors="coerce")

def _zip_as_str(s: pd.Series) -> pd.Series:
    """
    Cast zipcode to zero-padded 5-char strings where possible, avoiding
    accidental float-formatted values like "98103.0".
    """
    out = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    # Only pad when length <= 5; if some extended ZIP codes appear, keep them as-is.
    return out.str.zfill(5).where(out.str.len() <= 5, out)

def load_demographics(path: str = "data/zipcode_demographics.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "zipcode" not in df.columns:
        raise ValueError(f"'zipcode' column not found in {path}")
    df = df.copy()
    df["zipcode"] = _zip_as_str(df["zipcode"])
    return df

def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    # Light, safe imputations for different types
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            if df[c].isna().any():
                mode_val = df[c].mode(dropna=True)
                df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "")
    return df

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    # numeric, aligned Series (never scalars)
    yr_built       = _num_col(df, "yr_built",       CURRENT_YEAR)
    yr_renovated   = _num_col(df, "yr_renovated",   0)
    sqft_living    = _num_col(df, "sqft_living",    0)
    sqft_lot       = _num_col(df, "sqft_lot",       1)  # avoid /0
    sqft_basement  = _num_col(df, "sqft_basement",  0)
    floors         = _num_col(df, "floors",         0)

    # engineered features (now Series, so .clip/.fillna are fine)
    df["age_of_house"]      = (CURRENT_YEAR - yr_built).clip(lower=0).fillna(0)
    df["is_renovated"]      = (yr_renovated > 0).astype(int)
    df["living_to_lot_ratio"]= (sqft_living / sqft_lot.replace(0, 1)).fillna(0)
    df["basement_share"]    = (sqft_basement / sqft_living.replace(0, 1)).fillna(0)
    df["floors_x_living"]   = (floors * sqft_living).fillna(0)

    return df

def build_features(df_in: pd.DataFrame, demo_df: pd.DataFrame, add_engineered: bool = True) -> pd.DataFrame:
    """
    Given raw request columns (mirroring future_unseen_examples.csv),
    join zipcode demographics on the backend and perform light imputations.

    This function is the adapter from the external API contract to the
    internal model feature contract (together with feature ordering in
    model_features.json).
    """
    if "zipcode" not in df_in.columns:
        raise ValueError("Input payload must include 'zipcode'")

    df = df_in.copy()
    df["zipcode"] = _zip_as_str(df["zipcode"])

    merged = df.merge(demo_df, on="zipcode", how="left", suffixes=("", "_demo"))

    # track any missing demo rows (non-fatal)
    merged["_missing_demo"] = merged.filter(regex="_demo$|^ppltn_|^medn_|^hous_val_|^per_").isna().any(axis=1).astype(int)
    merged = _safe_fill(merged)

    if add_engineered:
        merged = _engineer(merged)
    
    return merged
