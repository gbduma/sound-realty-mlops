# training/create_model.py
from __future__ import annotations

import json
import pathlib
import pickle
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

SALES_PATH = "data/kc_house_data.csv"            # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved

# CV settings
N_SPLITS = 5
RANDOM_STATE = 42


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # guard against zeros/negatives
    eps = 1e-9
    y_true = np.maximum(eps, np.asarray(y_true))
    y_pred = np.maximum(eps, np.asarray(y_pred))
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Load the target and feature data by merging sales and demographics.

    Returns:
        x: DataFrame of features (after joining demographics), WITHOUT 'zipcode'
        y: Series target (price)
    """
    # Read sales (we'll also try to read 'date' for time-aware CV)
    sales_full = pd.read_csv(sales_path, dtype={'zipcode': str})
    # Subset to requested columns for modeling
    data = sales_full[sales_column_selection].copy()

    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price').astype(float)
    x = merged_data

    return x, y


def _build_pipeline() -> pipeline.Pipeline:
    """Keep the original baseline: RobustScaler + KNeighborsRegressor."""
    return pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        neighbors.KNeighborsRegressor()
    )


def _cross_validate(
    X: pd.DataFrame, y: pd.Series
) -> Dict[str, object]:
    """Run CV (KFold) and return metrics summary."""
    rmse_scores, rmsle_scores = [], []

    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = splitter.split(X)
    use_data = (X, y)

    for tr_idx, va_idx in splits:
        Xtr, Xv = use_data[0].iloc[tr_idx], use_data[0].iloc[va_idx]
        ytr, yv = use_data[1].iloc[tr_idx], use_data[1].iloc[va_idx]

        model = _build_pipeline()
        model.fit(Xtr, ytr)
        preds = model.predict(Xv)

        rmse_scores.append(_rmse(yv, preds))
        rmsle_scores.append(_rmsle(yv, preds))

    return {
        "cv_type": "KFold",
        "cv_folds": N_SPLITS,
        "rmse_per_fold": rmse_scores,
        "rmsle_per_fold": rmsle_scores,
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "rmsle_mean": float(np.mean(rmsle_scores)),
        "rmsle_std": float(np.std(rmsle_scores)),
    }

def _compute_permutation_importance(
    fitted_pipeline: pipeline.Pipeline,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Model-agnostic feature importance for the raw columns (works with KNN/tree-based models)."""
    try:
        # permutation importance returns importances aligned with input columns
        r = permutation_importance(
            fitted_pipeline, X_sample, y_sample,
            n_repeats=n_repeats, random_state=random_state, scoring=None
        )
        importances = {col: float(imp) for col, imp in zip(X_sample.columns, r.importances_mean)}
        # sort descending
        importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
        return importances
    except Exception:
        return {}


def main():
    """Load data, run CV, train final model, and export artifacts + metrics."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # Cross-validation on original baseline pipeline
    cv_stats = _cross_validate(x, y)

    # Fit final model on ALL data (as original script did on the train split)
    model = _build_pipeline().fit(x, y)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features (raw columns, pre-scaler)
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(output_dir / "model_features.json", "w", encoding="utf-8") as f:
        json.dump(list(x.columns), f)

    # Compute permutation importance on a small sample to keep runtime reasonable
    # Use last CV split or a simple tail sample if you prefer; here: 200-row sample
    sample_n = min(200, len(x))
    X_sample = x.tail(sample_n)
    y_sample = y.tail(sample_n)
    importances = _compute_permutation_importance(model, X_sample, y_sample, n_repeats=5)

    # Train-fit metrics on full data (optimistic by nature, CV is the proper generalization proxy)
    y_pred_full = model.predict(x)
    full_fit_metrics = {
        "rmse_full_fit": _rmse(y, y_pred_full),
        "rmsle_full_fit": _rmsle(y, y_pred_full),
    }

    # Write a compact model card / metadata file
    metrics = {
        "training_rows": int(len(x)),
        "feature_count": int(x.shape[1]),
        "cv": cv_stats,
        "metrics": full_fit_metrics,
        "permutation_importance": importances,  # may be empty if computation fails
        "notes": (
            "Baseline KNN with RobustScaler. CV gives a better sense of generalization than full-fit metrics. "
        ),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[train] wrote:", output_dir / "model.pkl", output_dir / "model_features.json", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
