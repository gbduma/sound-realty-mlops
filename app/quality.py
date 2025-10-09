from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

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
        return float(1 / (1 + np.exp(-raw)))  # squashed to (0,1)

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
