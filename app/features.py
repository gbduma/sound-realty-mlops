from __future__ import annotations
import pandas as pd
from pathlib import Path

def _zip_as_str(s: pd.Series) -> pd.Series:
    # robust cast to zero-padded 5-char strings where possible
    out = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    out = out.str.zfill(5).where(out.str.len() <= 5, out)  # avoid padding if already longer
    return out

def load_demographics(path: str = "data/zipcode_demographics.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "zipcode" not in df.columns:
        raise ValueError(f"'zipcode' column not found in {path}")
    df = df.copy()
    df["zipcode"] = _zip_as_str(df["zipcode"])
    return df

def build_features(df_in: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # Ensure zipcode is string for join
    if "zipcode" not in df.columns:
        raise ValueError("Input payload must include 'zipcode'")
    df["zipcode"] = _zip_as_str(df["zipcode"])

    # Join demographics (left join; keep request rows)
    merged = df.merge(demo_df, on="zipcode", how="left", suffixes=("", "_demo"))

    # Light numeric cleanup: fill NA for numeric cols with 0, others with mode
    for c in merged.columns:
        if pd.api.types.is_numeric_dtype(merged[c]):
            merged[c] = merged[c].fillna(0)
        else:
            if merged[c].isna().any():
                mode_val = merged[c].mode(dropna=True)
                merged[c] = merged[c].fillna(mode_val.iloc[0] if not mode_val.empty else "")

    return merged
