# training/create_model.py
from __future__ import annotations
import argparse, json, pathlib, pickle, numpy as np, pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import neighbors, pipeline, preprocessing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from app.features import load_demographics, build_features

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
N_ITER = 25  # for RandomizedSearchCV

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # guard against zeros/negatives
    eps = 1e-9
    y_true = np.maximum(eps, np.asarray(y_true))
    y_pred = np.maximum(eps, np.asarray(y_pred))
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def _cv(model, X, y, folds=N_SPLITS):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rmses, rmsles = [], []
    for tr, va in kf.split(X):
        Xtr, Xv, ytr, yv = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        rmses.append(_rmse(yv, pred)); rmsles.append(_rmsle(yv, pred))
    return {"rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "rmsle_mean": float(np.mean(rmsles)), "rmsle_std": float(np.std(rmsles))}

def _save_artifacts(out_dir: pathlib.Path, model, feature_order, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f: pickle.dump(model, f)
    with open(out_dir / "model_features.json", "w", encoding="utf-8") as f: json.dump(list(feature_order), f)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2)

# ---------- Baseline recipe ----------
def train_baseline_knn() -> pathlib.Path:
    sales = pd.read_csv(SALES_PATH, dtype={"zipcode": str})
    data = sales[[c for c in SALES_COLUMN_SELECTION if c in sales.columns]].copy()
    demos = load_demographics(DEMOGRAPHICS_PATH)

    # baseline: join demographics, NO engineered features
    merged = build_features(data, demos, add_engineered=False)
    y = merged.pop("price").astype(float)
    X = merged.select_dtypes(include=["number"]).copy()  # numeric only; zipcode (str) excluded

    knn = pipeline.make_pipeline(preprocessing.RobustScaler(), neighbors.KNeighborsRegressor())
    cv_stats = _cv(knn, X, y)
    knn.fit(X, y)

    out = pathlib.Path(OUTPUT_DIR) / "baseline_knn"
    metrics = {
        "training_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "cv": cv_stats,
        "notes": "Baseline per original script: RobustScaler + KNN, sales selection + demographics, no engineered features."
    }
    _save_artifacts(out, knn, X.columns, metrics)
    print("[train] baseline_knn wrote:", out)
    return out

# ---------- Enhanced recipe ----------
def _random_search(model, params, X, y, n_iter=N_ITER):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(model, params, n_iter=n_iter, scoring="neg_root_mean_squared_error",
                            cv=kf, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
    rs.fit(X, y)
    best = rs.best_estimator_
    scores = _cv(best, X, y)
    return best, scores, {"best_params": rs.best_params_}

def train_enhanced_auto() -> pathlib.Path:
    sales = pd.read_csv(SALES_PATH, dtype={"zipcode": str})
    # enhanced: ALL reasonable sales cols (everything except id/date/price) + demographics + engineered
    drop_cols = {"price", "date", "id"}
    base_cols = [c for c in sales.columns if c not in drop_cols]
    data = sales[base_cols].copy()
    data["price"] = sales["price"].astype(float)  # attach target so build_features can pop it later if you prefer

    demos = load_demographics(DEMOGRAPHICS_PATH)
    merged = build_features(data, demos, add_engineered=True)
    y = merged.pop("price").astype(float)
    X = merged.select_dtypes(include=["number"]).copy()

    # candidates
    models = []
    # KNN baseline (for reference in leaderboard)
    knn = pipeline.make_pipeline(preprocessing.RobustScaler(), neighbors.KNeighborsRegressor())
    models.append(("KNN+RobustScaler", knn, _cv(knn, X, y), None))

    # RandomForest sweep
    rf = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    rf_params = {"n_estimators":[200,400,800], "max_depth":[None,10,20,40], "min_samples_leaf":[1,2,5]}
    rf_best, rf_scores, rf_info = _random_search(rf, rf_params, X, y); models.append(("RandomForest", rf_best, rf_scores, rf_info))

    # XGB sweep
    xgb = XGBRegressor(n_estimators=500, max_depth=6, subsample=0.8, colsample_bytree=0.8, learning_rate=0.05,
                       random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    xgb_params = {"max_depth":[4,6,8], "subsample":[0.7,0.9], "colsample_bytree":[0.7,0.9], "learning_rate":[0.03,0.05,0.1], "n_estimators":[400,600,800]}
    xgb_best, xgb_scores, xgb_info = _random_search(xgb, xgb_params, X, y); models.append(("XGBoost", xgb_best, xgb_scores, xgb_info))

    # LGBM sweep
    lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE)
    lgbm_params = {"num_leaves":[31,63], "learning_rate":[0.03,0.05,0.1], "n_estimators":[600,800,1000], "feature_fraction":[0.7,0.9], "bagging_fraction":[0.7,0.9]}
    lgbm_best, lgbm_scores, lgbm_info = _random_search(lgbm, lgbm_params, X, y); models.append(("LightGBM", lgbm_best, lgbm_scores, lgbm_info))

    # leaderboard + winner
    leaderboard = [{"model": name, **scores, **(info or {})} for name, _, scores, info in models]
    winner_name, winner_model = min(models, key=lambda t: t[2]["rmse_mean"])[0], min(models, key=lambda t: t[2]["rmse_mean"])[1]
    winner_model.fit(X, y)

    out = pathlib.Path(OUTPUT_DIR) / "enhanced_auto"
    full_fit = {"rmse_full_fit": _rmse(y, winner_model.predict(X)), "rmsle_full_fit": _rmsle(y, winner_model.predict(X))}
    metrics = {
        "training_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "leaderboard": leaderboard,
        "winner": {"name": winner_name, "rmse_mean": min(leaderboard, key=lambda r: r["rmse_mean"])["rmse_mean"]},
        "metrics_full_fit": full_fit,
        "notes": "Enhanced: demographics + engineered features; light randomized sweeps across RF/XGB/LGBM."
    }
    _save_artifacts(out, winner_model, X.columns, metrics)
    print("[train] enhanced_auto wrote:", out)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=["baseline_knn","enhanced_auto","both"], default="both")
    args = parser.parse_args()

    if args.recipe in ("baseline_knn","both"): train_baseline_knn()
    if args.recipe in ("enhanced_auto","both"): train_enhanced_auto()

if __name__ == "__main__":
    main()