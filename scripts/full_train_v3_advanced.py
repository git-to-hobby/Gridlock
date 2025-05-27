#!/usr/bin/env python3
"""Full-train v3: tail-aware advanced pipeline.

Integrates on top of full_train_v2 feature base:
  1. Asymmetric loss (LGBM custom objective + CatBoost sample weights)
  2. K-Fold target encoding with smoothing (geohash, gh_p4, RoadType, interactions)
  3. Lag & rolling features (t-1, t-2, roll-3, roll-6) per geohash timeline
  4. Stratified CV by demand bins (preserves y=1.0 extremes in each fold)

Blends with proven LB anchor (submission_9059) and optionally push_triple_w145 stack.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import r2_score

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from full_train_v2 import (  # noqa: E402
    add_latlon,
    add_time,
    attach_day48_features,
    impute_temperature_group_medians,
)
from src.advanced.asymmetric_loss import (  # noqa: E402
    asymmetric_sample_weights,
    lgbm_asymmetric_objective,
)
from src.advanced.lag_features import add_lag_rolling_features  # noqa: E402
from src.advanced.stratified_cv import assign_fold_ids, stratified_kfold_indices  # noqa: E402
from src.advanced.target_encoding import kfold_target_encode, multi_col_target_encode  # noqa: E402

DATA_DIR = REPO_ROOT / "dataset"
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)


def build_features_v3():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train = add_time(train)
    test = add_time(test)
    for frame in (train, test):
        for c in ("RoadType", "Weather", "LargeVehicles", "Landmarks"):
            frame[c] = frame[c].fillna("Missing")
    train, test = impute_temperature_group_medians(train, test)
    train = add_latlon(train)
    test = add_latlon(test)
    day48 = train[train["day"] == 48].copy()
    train = attach_day48_features(train, day48)
    test = attach_day48_features(test, day48)

    # Lag / rolling (uses train history for test rows)
    train, test, lag_cols = add_lag_rolling_features(train, test)

    y = train["demand"].to_numpy()
    fold_id = assign_fold_ids(y, n_splits=5, random_state=2026)

    # K-Fold TE (replaces LOO — aligned with stratified folds)
    train, test, te_cols = kfold_target_encode(
        train, test,
        cols=["geohash", "gh_p4", "RoadType", "Weather"],
        smoothing=20.0,
        n_splits=5,
        fold_id=fold_id,
    )
    te_key_cols = ["geohash_slot", "geohash_hour", "gh_p4_slot"]
    train, test, te_ix_cols = multi_col_target_encode(
        train, test,
        interaction_cols=[["geohash", "slot"], ["geohash", "hour"], ["gh_p4", "slot"]],
        smoothing=15.0,
        n_splits=5,
        fold_id=fold_id,
    )
    te_cols = te_cols + te_ix_cols
    train = train.drop(columns=te_key_cols, errors="ignore")
    test = test.drop(columns=te_key_cols, errors="ignore")

    cat_features = ["geohash", "gh_p3", "gh_p4", "gh_p5", "RoadType", "Weather", "LargeVehicles", "Landmarks"]
    drop_cols = {"Index", "demand", "timestamp", *te_key_cols}
    feature_cols = [c for c in train.columns if c not in drop_cols]
    for col in cat_features:
        train[col] = train[col].astype(str).fillna("Missing")
        test[col] = test[col].astype(str).fillna("Missing")
    for col in feature_cols:
        if col in cat_features:
            continue
        train[col] = pd.to_numeric(train[col], errors="coerce").astype(float).fillna(-1.0)
        test[col] = pd.to_numeric(test[col], errors="coerce").astype(float).fillna(-1.0)

    return train, test, feature_cols, cat_features, y, fold_id


def train_catboost_v3(train, test, feature_cols, cat_features, y, fold_id):
    cat_idx = [feature_cols.index(c) for c in cat_features]
    X, X_test = train[feature_cols], test[feature_cols]
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    splits = stratified_kfold_indices(y, n_splits=5, random_state=2026)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        t0 = time.time()
        sw = asymmetric_sample_weights(y[tr_idx])
        model = CatBoostRegressor(
            iterations=2200,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=5,
            random_seed=2026 + fold,
            verbose=0,
            allow_writing_files=False,
            early_stopping_rounds=100,
        )
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_idx, weight=sw)
        valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_idx)
        model.fit(train_pool, eval_set=valid_pool)
        oof[va_idx] = np.clip(model.predict(X.iloc[va_idx]), 0, 1)
        test_pred += np.clip(model.predict(X_test), 0, 1) / len(splits)
        ext = int((y[va_idx] >= 0.99).sum())
        print(
            f"  CatBoost-v3 fold {fold}: R2={r2_score(y[va_idx], oof[va_idx]):.6f}  "
            f"extremes_in_va={ext}  ({time.time()-t0:.0f}s)",
            flush=True,
        )
    tail_mask = y >= 0.99
    tail_r2 = r2_score(y[tail_mask], oof[tail_mask]) if tail_mask.sum() > 10 else float("nan")
    print(f"  CatBoost-v3 OOF={r2_score(y,oof):.6f}  tail(y>=0.99) R2={tail_r2:.4f}", flush=True)
    return test_pred, oof


def train_lightgbm_v3(train, test, feature_cols, cat_features, y, fold_id):
    train_l = train.copy()
    test_l = test.copy()
    for col in cat_features:
        joint = pd.concat([train_l[col], test_l[col]], axis=0).astype(str)
        mapping = {v: i for i, v in enumerate(sorted(joint.unique()))}
        train_l[col] = train_l[col].astype(str).map(mapping).astype(int)
        test_l[col] = test_l[col].astype(str).map(mapping).astype(int)

    X, X_test = train_l[feature_cols], test_l[feature_cols]
    oof = np.zeros(len(train_l))
    test_pred = np.zeros(len(test_l))
    obj, feval = lgbm_asymmetric_objective(alpha=3.0, high_y=0.80)
    splits = stratified_kfold_indices(y, n_splits=5, random_state=2026)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        t0 = time.time()
        model = LGBMRegressor(
            objective=obj,
            n_estimators=2500,
            learning_rate=0.025,
            max_depth=-1,
            num_leaves=127,
            min_child_samples=10,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=4051 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric=feval,
            categorical_feature=cat_features,
            callbacks=[early_stopping(100, verbose=False)],
        )
        oof[va_idx] = np.clip(model.predict(X.iloc[va_idx]), 0, 1)
        test_pred += np.clip(model.predict(X_test), 0, 1) / len(splits)
        print(f"  LightGBM-v3 fold {fold}: R2={r2_score(y[va_idx], oof[va_idx]):.6f}  ({time.time()-t0:.0f}s)", flush=True)

    tail_mask = y >= 0.99
    tail_r2 = r2_score(y[tail_mask], oof[tail_mask]) if tail_mask.sum() > 10 else float("nan")
    print(f"  LightGBM-v3 OOF={r2_score(y,oof):.6f}  tail(y>=0.99) R2={tail_r2:.4f}", flush=True)
    return test_pred, oof


def extreme_tail_correction(pred: np.ndarray, train_y: np.ndarray, quantile: float = 0.995) -> np.ndarray:
    """Map top-quantile predictions toward 1.0 using train extreme rate (gentle)."""
    p = pred.copy()
    q = float(np.quantile(p, quantile))
    train_ext_rate = float((train_y >= 0.99).mean())
    pred_ext_rate = float((p >= q).mean())
    if pred_ext_rate > 0 and train_ext_rate > pred_ext_rate:
        # Gentle stretch for top bin only
        mask = p >= q
        p[mask] = np.minimum(1.0, p[mask] + 0.02 * (1.0 - p[mask]))
    return np.clip(p, 0, 1)


def main():
    print("[1/4] Build v3 features (lag, rolling, KFold TE, stratified folds)...", flush=True)
    train, test, feature_cols, cat_features, y, fold_id = build_features_v3()
    n_ext = int((y >= 0.99).sum())
    print(f"  features={len(feature_cols)}  train extremes(y>=0.99)={n_ext}", flush=True)

    print("[2/4] CatBoost v3 (asymmetric weights)...", flush=True)
    cat_pred, _ = train_catboost_v3(train, test, feature_cols, cat_features, y, fold_id)

    print("[3/4] LightGBM v3 (asymmetric objective)...", flush=True)
    lgbm_pred, _ = train_lightgbm_v3(train, test, feature_cols, cat_features, y, fold_id)

    print("[4/4] Compose submissions...", flush=True)
    blend = 0.5 * cat_pred + 0.5 * lgbm_pred
    blend = extreme_tail_correction(blend, y)

    def write(name, arr):
        out = pd.DataFrame({"Index": test["Index"].astype(int), "demand": np.clip(arr, 0, 1)})
        out = out.sort_values("Index")
        out.to_csv(OUT_DIR / name, index=False, float_format="%.16g", lineterminator="\n")
        print(f"  {name:<42} mean={out['demand'].mean():.5f}", flush=True)

    base_df = pd.read_csv(OUT_DIR / "submission_9059.csv").sort_values("Index")
    w145_path = OUT_DIR / "push_triple_w145.csv"
    w145_df = pd.read_csv(w145_path).sort_values("Index") if w145_path.exists() else None

    write("v3adv_catboost.csv", cat_pred)
    write("v3adv_lgbm.csv", lgbm_pred)
    write("v3adv_blend.csv", blend)

    for bw in [0.40, 0.45, 0.50]:
        safe = pd.DataFrame({"Index": test["Index"].astype(int), "demand": blend})
        safe = safe.merge(base_df[["Index", "demand"]], on="Index", suffixes=("_new", "_base"))
        safe["demand"] = bw * safe["demand_base"] + (1 - bw) * safe["demand_new"]
        safe[["Index", "demand"]].sort_values("Index").to_csv(
            OUT_DIR / f"v3adv_safe{int(bw*100)}.csv", index=False, float_format="%.16g", lineterminator="\n"
        )
        print(f"  v3adv_safe{int(bw*100)}.csv{'':28} mean={safe['demand'].mean():.5f}", flush=True)

    if w145_df is not None:
        for aw in [0.85, 0.90, 0.95]:
            mix = pd.DataFrame({"Index": test["Index"].astype(int), "demand": blend})
            mix = mix.merge(w145_df[["Index", "demand"]], on="Index", suffixes=("_new", "_w145"))
            mix["demand"] = aw * mix["demand_w145"] + (1 - aw) * mix["demand_new"]
            mix[["Index", "demand"]].sort_values("Index").to_csv(
                OUT_DIR / f"v3adv_w145mix_{int(aw*100)}.csv", index=False, float_format="%.16g", lineterminator="\n"
            )
            print(f"  v3adv_w145mix_{int(aw*100)}.csv{'':22} mean={mix['demand'].mean():.5f}", flush=True)

    print("Done. Start with v3adv_safe45.csv or v3adv_w145mix_90.csv", flush=True)


if __name__ == "__main__":
    main()
