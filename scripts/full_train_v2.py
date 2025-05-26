#!/usr/bin/env python3
"""Full-train v2 with competitor's reported unlocks:
1. Explicit geohash x slot target encoding (leakage-safe, leave-one-out style).
2. Decoded lat/lon from geohash as continuous spatial features.
3. Group-median Temperature imputation (gh_p4 + day, then gh_p4, then global).
4. LightGBM 2000 trees + CatBoost native categorical.
5. Apply proven LB tail boost on top.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from spatial_features import decode_geohash

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DATA_DIR = REPO_ROOT / "dataset"
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["timestamp"].astype(str).str.split(":", expand=True).astype(int)
    out["hour"] = parts[0]
    out["minute"] = parts[1]
    out["slot"] = out["hour"] * 4 + out["minute"] // 15
    out["slot_sin"] = np.sin(2 * np.pi * out["slot"] / 96)
    out["slot_cos"] = np.cos(2 * np.pi * out["slot"] / 96)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60)
    for precision in (3, 4, 5):
        out[f"gh_p{precision}"] = out["geohash"].str[:precision]
    return out


def add_latlon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    coords = out["geohash"].map(lambda g: decode_geohash(g)[:2])
    out["gh_lat"] = coords.map(lambda x: x[0])
    out["gh_lon"] = coords.map(lambda x: x[1])
    centroid_lat = out["gh_lat"].mean()
    centroid_lon = out["gh_lon"].mean()
    out["gh_dlat"] = out["gh_lat"] - centroid_lat
    out["gh_dlon"] = out["gh_lon"] - centroid_lon
    out["gh_dist_centroid"] = np.sqrt(out["gh_dlat"] ** 2 + out["gh_dlon"] ** 2)
    return out


def impute_temperature_group_medians(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    combined = pd.concat([train.assign(_origin="train"), test.assign(_origin="test")], ignore_index=True)
    by_p4_day = combined.groupby(["gh_p4", "day"])["Temperature"].transform("median")
    by_p4 = combined.groupby("gh_p4")["Temperature"].transform("median")
    global_median = combined["Temperature"].median()
    combined["Temperature_imp"] = (
        combined["Temperature"]
        .fillna(by_p4_day)
        .fillna(by_p4)
        .fillna(global_median)
    )
    train["Temperature"] = combined.loc[combined["_origin"] == "train", "Temperature_imp"].to_numpy()
    test["Temperature"] = combined.loc[combined["_origin"] == "test", "Temperature_imp"].to_numpy()
    return train, test


def attach_day48_features(frame: pd.DataFrame, day48: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    same = day48[["geohash", "slot", "demand"]].rename(columns={"demand": "d48_same_slot"})
    out = out.merge(same, on=["geohash", "slot"], how="left")
    gh_stats = day48.groupby("geohash")["demand"].agg(["mean", "std", "min", "max"]).rename(
        columns=lambda c: f"d48_gh_{c}"
    )
    out = out.merge(gh_stats, left_on="geohash", right_index=True, how="left")
    p4_stats = day48.groupby("gh_p4")["demand"].agg(["mean", "std"]).rename(columns=lambda c: f"d48_p4_{c}")
    out = out.merge(p4_stats, left_on="gh_p4", right_index=True, how="left")
    slot_stats = day48.groupby("slot")["demand"].agg(["mean", "std"]).rename(columns=lambda c: f"d48_slot_{c}")
    out = out.merge(slot_stats, left_on="slot", right_index=True, how="left")
    for offset in (-4, -2, -1, 1, 2, 4, 8):
        shifted = day48[["geohash", "slot", "demand"]].copy()
        shifted["slot"] = shifted["slot"] - offset
        shifted = shifted.rename(columns={"demand": f"d48_slot_{offset:+d}"})
        out = out.merge(shifted, on=["geohash", "slot"], how="left")
    return out


def gh_slot_target_encoding_loo(train: pd.DataFrame, test: pd.DataFrame, smoothing: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-out (geohash, slot) target encoding for train.
    Test uses the full (geohash, slot) mean computed from train.
    """
    train = train.copy()
    test = test.copy()
    global_mean = float(train["demand"].mean())

    grp = train.groupby(["geohash", "slot"])["demand"]
    gh_slot_sum = grp.transform("sum")
    gh_slot_count = grp.transform("count")
    train["gh_slot_te"] = (
        (gh_slot_sum - train["demand"] + smoothing * global_mean)
        / (gh_slot_count - 1 + smoothing)
    )

    map_df = train.groupby(["geohash", "slot"]).agg(
        gh_slot_mean=("demand", "mean"),
        gh_slot_count=("demand", "count"),
    )
    map_df["gh_slot_te_full"] = (
        map_df["gh_slot_count"] * map_df["gh_slot_mean"] + smoothing * global_mean
    ) / (map_df["gh_slot_count"] + smoothing)
    test_idx = pd.MultiIndex.from_frame(test[["geohash", "slot"]])
    test["gh_slot_te"] = test_idx.map(map_df["gh_slot_te_full"]).astype(float)
    test["gh_slot_te"] = test["gh_slot_te"].fillna(global_mean)
    train["gh_slot_te"] = train["gh_slot_te"].fillna(global_mean)
    return train, test


def gh_hour_target_encoding_loo(train: pd.DataFrame, test: pd.DataFrame, smoothing: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    global_mean = float(train["demand"].mean())
    grp = train.groupby(["geohash", "hour"])["demand"]
    gh_hour_sum = grp.transform("sum")
    gh_hour_count = grp.transform("count")
    train["gh_hour_te"] = (
        (gh_hour_sum - train["demand"] + smoothing * global_mean)
        / (gh_hour_count - 1 + smoothing)
    )
    map_df = train.groupby(["geohash", "hour"]).agg(
        gh_hour_mean=("demand", "mean"),
        gh_hour_count=("demand", "count"),
    )
    map_df["gh_hour_te_full"] = (
        map_df["gh_hour_count"] * map_df["gh_hour_mean"] + smoothing * global_mean
    ) / (map_df["gh_hour_count"] + smoothing)
    test_idx = pd.MultiIndex.from_frame(test[["geohash", "hour"]])
    test["gh_hour_te"] = test_idx.map(map_df["gh_hour_te_full"]).astype(float)
    test["gh_hour_te"] = test["gh_hour_te"].fillna(global_mean)
    train["gh_hour_te"] = train["gh_hour_te"].fillna(global_mean)
    return train, test


def build_features():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train = add_time(train)
    test = add_time(test)
    for frame in (train, test):
        frame["RoadType"] = frame["RoadType"].fillna("Missing")
        frame["Weather"] = frame["Weather"].fillna("Missing")
        frame["LargeVehicles"] = frame["LargeVehicles"].fillna("Missing")
        frame["Landmarks"] = frame["Landmarks"].fillna("Missing")
    train, test = impute_temperature_group_medians(train, test)
    train = add_latlon(train)
    test = add_latlon(test)
    day48 = train[train["day"] == 48].copy()
    train = attach_day48_features(train, day48)
    test = attach_day48_features(test, day48)
    train, test = gh_slot_target_encoding_loo(train, test, smoothing=10.0)
    train, test = gh_hour_target_encoding_loo(train, test, smoothing=10.0)

    cat_features = ["geohash", "gh_p3", "gh_p4", "gh_p5", "RoadType", "Weather", "LargeVehicles", "Landmarks"]
    drop_cols = {"Index", "demand", "timestamp"}
    feature_cols = [c for c in train.columns if c not in drop_cols]
    for col in cat_features:
        train[col] = train[col].astype(str).fillna("Missing")
        test[col] = test[col].astype(str).fillna("Missing")
    for col in feature_cols:
        if col in cat_features:
            continue
        train[col] = pd.to_numeric(train[col], errors="coerce").astype(float)
        test[col] = pd.to_numeric(test[col], errors="coerce").astype(float)
        train[col] = train[col].fillna(-1.0)
        test[col] = test[col].fillna(-1.0)
    return train, test, feature_cols, cat_features


def train_catboost(train, test, feature_cols, cat_features):
    cat_idx = [feature_cols.index(c) for c in cat_features]
    y = train["demand"].to_numpy()
    X = train[feature_cols]
    X_test = test[feature_cols]
    kf = KFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        t0 = time.time()
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
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_idx)
        valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_idx)
        model.fit(train_pool, eval_set=valid_pool)
        oof[va_idx] = np.clip(model.predict(X.iloc[va_idx]), 0, 1)
        test_pred += np.clip(model.predict(X_test), 0, 1) / kf.n_splits
        print(f"  CatBoost fold {fold}: R2={r2_score(y[va_idx], oof[va_idx]):.6f}  ({time.time()-t0:.1f}s)", flush=True)
    return test_pred, float(r2_score(y, oof))


def train_lightgbm(train, test, feature_cols, cat_features):
    train_l = train.copy()
    test_l = test.copy()
    for col in cat_features:
        joint = pd.concat([train_l[col], test_l[col]], axis=0).astype(str)
        mapping = {v: i for i, v in enumerate(sorted(joint.unique()))}
        train_l[col] = train_l[col].astype(str).map(mapping).astype(int)
        test_l[col] = test_l[col].astype(str).map(mapping).astype(int)
    y = train_l["demand"].to_numpy()
    X = train_l[feature_cols]
    X_test = test_l[feature_cols]
    kf = KFold(n_splits=5, shuffle=True, random_state=4051)
    oof = np.zeros(len(train_l))
    test_pred = np.zeros(len(test_l))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        t0 = time.time()
        model = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=127,
            min_child_samples=15,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.5,
            random_state=4051 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            categorical_feature=cat_features,
            callbacks=[],
        )
        oof[va_idx] = np.clip(model.predict(X.iloc[va_idx]), 0, 1)
        test_pred += np.clip(model.predict(X_test), 0, 1) / kf.n_splits
        print(f"  LightGBM fold {fold}: R2={r2_score(y[va_idx], oof[va_idx]):.6f}  ({time.time()-t0:.1f}s)", flush=True)
    return test_pred, float(r2_score(y, oof))


def apply_tail_boost(pred: np.ndarray, b1: float = 1.075, b2: float = 1.035) -> np.ndarray:
    p = pred.copy()
    q1 = float(np.quantile(p, 0.965))
    q2 = float(np.quantile(p, 0.98))
    p[p > q1] *= b1
    p[p > q2] *= b2
    return np.clip(p, 0, 1)


def main() -> None:
    print("[1/5] Building features (group-imputed temp, lat/lon, TE)...", flush=True)
    train, test, feature_cols, cat_features = build_features()
    print(f"  features={len(feature_cols)} including {len(cat_features)} categorical", flush=True)
    print(f"  feature list: {feature_cols}", flush=True)

    print("[2/5] CatBoost (5-fold CV)...", flush=True)
    cat_pred, cat_oof = train_catboost(train, test, feature_cols, cat_features)
    print(f"  CatBoost OOF = {cat_oof:.6f}", flush=True)

    print("[3/5] LightGBM (5-fold CV)...", flush=True)
    lgbm_pred, lgbm_oof = train_lightgbm(train, test, feature_cols, cat_features)
    print(f"  LightGBM OOF = {lgbm_oof:.6f}", flush=True)

    print("[4/5] Composing v2 candidates...", flush=True)
    base_df = pd.read_csv(OUT_DIR / "submission_9059.csv")
    base = base_df["demand"].to_numpy()
    new_blend = 0.5 * cat_pred + 0.5 * lgbm_pred

    def write(name: str, arr: np.ndarray) -> None:
        out = pd.DataFrame({"Index": base_df["Index"].astype(int), "demand": np.clip(arr, 0, 1)})
        out.to_csv(OUT_DIR / name, index=False)
        print(f"  {name:<48} mean={out['demand'].mean():.6f}  std={out['demand'].std():.6f}", flush=True)

    write("v2full_catboost_only.csv", cat_pred)
    write("v2full_lgbm_only.csv", lgbm_pred)
    write("v2full_cat_lgbm_blend.csv", new_blend)
    write("v2full_cat_lgbm_blend_boost.csv", apply_tail_boost(new_blend))

    for bw in [0.45, 0.50, 0.55, 0.60, 0.65]:
        blend = bw * base + (1 - bw) * new_blend
        write(f"v2full_safe{int(bw*100)}_blend.csv", blend)
        write(f"v2full_safe{int(bw*100)}_blend_boost.csv", apply_tail_boost(blend))

    print("[5/5] Done.", flush=True)


if __name__ == "__main__":
    main()
