from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def _sigmoid01(raw) -> float:
    """
    Return a (0,1) score. Accepts scalar or ndarray. If an array is passed,
    reduce to a scalar (mean) before applying sigmoid to avoid deprecation.
    """
    arr = np.asarray(raw, dtype=float)
    score = arr.item() if arr.size == 1 else float(arr.ravel().mean())
    # score is now a plain float, not an array
    return float(1.0 / (1.0 + np.exp(-score)))

class OODGuard:
    def __init__(self, X_train: pd.DataFrame):
        # numeric only for OOD
        num = X_train.select_dtypes(include=["number"]).fillna(0)
        self.cols = list(num.columns)
        self.clf = IsolationForest(n_estimators=300, random_state=42, contamination="auto")
        self.clf.fit(num.values)

    def score(self, row: Dict[str, Any]) -> float:
        # returns anomaly score in [0,1], where 1 is very typical
        x = pd.DataFrame([row]).reindex(columns=self.cols).fillna(0)
        raw = self.clf.score_samples(x.values)  # higher is more normal
        return _sigmoid01(raw)  # squashed to (0,1)

def basic_rules(row: Dict[str, Any]) -> Dict[str, Any]:
    msgs = []
    def bad(cond, msg): 
        if cond: msgs.append(msg)
    b = float(row.get("bathrooms", 0))
    br= float(row.get("bedrooms", 0))
    sl= float(row.get("sqft_living", 0))
    zl= float(row.get("sqft_lot", 0))
    bad(b < 0 or b > 10, "bathrooms out of range")
    bad(br < 0 or br > 10, "bedrooms out of range")
    bad(sl < 200 or sl > 20000, "sqft_living out of range")
    bad(zl < 200 or zl > 1000000, "sqft_lot out of range")
    return {"messages": msgs, "penalty": 0.15*len(msgs)}
