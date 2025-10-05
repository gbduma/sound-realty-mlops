# app/features.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


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


def build_features(df_in: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
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

    # Light, safe imputations
    for c in merged.columns:
        if pd.api.types.is_numeric_dtype(merged[c]):
            merged[c] = merged[c].fillna(0)
        else:
            if merged[c].isna().any():
                mode_val = merged[c].mode(dropna=True)
                merged[c] = merged[c].fillna(mode_val.iloc[0] if not mode_val.empty else "")

    # place for additional engineered features later, e.g.:
    # merged["age_of_house"] = current_year - merged["yr_built"]
    return merged
