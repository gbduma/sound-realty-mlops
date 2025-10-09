from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd

EARTH_R = 6371.0  # km

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

@dataclass
class CompsConfig:
    max_km: float = 5.0
    k_neighbors: int = 25
    min_neighbors: int = 5

class CompsEstimator:
    def __init__(self, sales_df: pd.DataFrame, config: CompsConfig = CompsConfig()):
        # expects columns: price, lat, long, bedrooms, bathrooms, sqft_living, sqft_lot, floors, yr_built
        self.cfg = config
        self.df = sales_df.dropna(subset=["price","lat","long"]).copy()
        for c in ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","yr_built"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        self.df = self.df.dropna()

    def estimate(self, row: dict) -> dict:
        try:
            lat, lon = float(row.get("lat", np.nan)), float(row.get("long", np.nan))
            if np.isnan(lat) or np.isnan(lon):
                return {"comp_count": 0}

            d = haversine_km(lat, lon, self.df["lat"].values, self.df["long"].values)
            nearby = self.df.loc[d <= self.cfg.max_km].copy()
            nearby["dist_km"] = d[d <= self.cfg.max_km]
            if nearby.empty:
                return {"comp_count": 0}

            # simple similarity score on core attributes
            # --- robust z-score helpers (median / MAD) ---
            def _robust_center_scale(s: pd.Series):
                med = np.median(s.values)
                mad = np.median(np.abs(s.values - med))
                scale = mad if mad > 1e-9 else np.std(s.values) + 1e-9  # fallback if mad ~ 0
                return med, scale

            sim_cols = []
            for c in ["bedrooms", "bathrooms", "sqft_living", "yr_built"]:
                if c in nearby.columns and c in row:
                    s = pd.to_numeric(nearby[c], errors="coerce").fillna(0)
                    med, scale = _robust_center_scale(s)
                    z_series = (s - med) / scale
                    z_row = (float(row.get(c, 0)) - med) / scale
                    diff2 = (z_series - z_row) ** 2
                    col = f"sim_{c}"
                    nearby[col] = diff2
                    sim_cols.append(col)

            # rank by distance + similarity
            w_dist = 1.0
            w_sim  = 0.05
            nearby["rank"] = w_dist * nearby["dist_km"] + w_sim * (nearby[sim_cols].sum(axis=1) if sim_cols else 0.0)
            comps = nearby.nsmallest(self.cfg.k_neighbors, "rank")
            n = len(comps)
            if n < self.cfg.min_neighbors:
                return {"comp_count": int(n)}

            # robust aggregate
            est = float(comps["price"].median())
            lo  = float(comps["price"].quantile(0.05))
            hi  = float(comps["price"].quantile(0.95))
            return {
                "comp_count": int(n),
                "comps_estimate": est,
                "comps_band": [lo, hi],
                "avg_dist_km": float(comps["dist_km"].mean())
            }
        except Exception:
            return {"comp_count": 0}
