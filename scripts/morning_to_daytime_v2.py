#!/usr/bin/env python3
"""Enhanced morning->daytime forecaster: CatBoost + LightGBM ensemble, richer features."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DATA_DIR = REPO_ROOT / "dataset"
OUT_DIR = REPO_ROOT / "output"
MORNING_SLOTS = list(range(0, 9))


def add_time(df):
    out = df.copy()
    p = out["timestamp"].astype(str).str.split(":", expand=True).astype(int)
    out["hour"] = p[0]; out["minute"] = p[1]
    out["slot"] = p[0] * 4 + p[1] // 15
    out["slot_sin"] = np.sin(2 * np.pi * out["slot"] / 96)
    out["slot_cos"] = np.cos(2 * np.pi * out["slot"] / 96)
    for pr in (3, 4):
        out[f"gh_p{pr}"] = out["geohash"].str[:pr]
    return out


def morning_profile(day_df):
    m = day_df[day_df["slot"].isin(MORNING_SLOTS)]
    piv = m.pivot_table(index="geohash", columns="slot", values="demand").reindex(columns=MORNING_SLOTS)
    f = pd.DataFrame(index=piv.index)
    for s in MORNING_SLOTS:
        f[f"m{s}"] = piv[s]
    f["m_mean"] = piv.mean(axis=1); f["m_std"] = piv.std(axis=1)
    f["m_max"] = piv.max(axis=1); f["m_min"] = piv.min(axis=1)
    f["m_last"] = piv[8]; f["m_first"] = piv[0]
    f["m_last3"] = piv[[6, 7, 8]].mean(axis=1)
    f["m_first3"] = piv[[0, 1, 2]].mean(axis=1)
    f["m_momentum"] = f["m_last3"] - f["m_first3"]
    f["m_accel"] = (piv[8] - piv[7]) - (piv[1] - piv[0])
    x = np.array(MORNING_SLOTS, float); xc = x - x.mean(); denom = (xc ** 2).sum()
    vals = piv.to_numpy()
    f["m_trend"] = np.nansum((vals - np.nanmean(vals, axis=1, keepdims=True)) * xc, axis=1) / denom
    f["m_range"] = f["m_max"] - f["m_min"]
    f["m_last_over_mean"] = (f["m_last"] / f["m_mean"].replace(0, np.nan)).clip(0, 5)
    return f.reset_index()


def build():
    raw = add_time(pd.read_csv(DATA_DIR / "train.csv"))
    test = add_time(pd.read_csv(DATA_DIR / "test.csv"))
    for f in (raw, test):
        for c in ("RoadType", "Weather", "LargeVehicles", "Landmarks"):
            f[c] = f[c].fillna("Missing")
        f["Temperature"] = f["Temperature"].fillna(raw["Temperature"].median())
    d48 = raw[raw["day"] == 48]; d49 = raw[raw["day"] == 49]
    tr = d48[d48["slot"].between(9, 55)].merge(morning_profile(d48), on="geohash", how="left")
    te = test.merge(morning_profile(d49), on="geohash", how="left")
    morning_cols = [f"m{s}" for s in MORNING_SLOTS] + [
        "m_mean", "m_std", "m_max", "m_min", "m_last", "m_first", "m_last3", "m_first3",
        "m_momentum", "m_accel", "m_trend", "m_range", "m_last_over_mean"]
    static_cols = ["slot", "hour", "slot_sin", "slot_cos", "Temperature", "NumberofLanes"]
    cat_cols = ["geohash", "gh_p4", "RoadType", "Weather", "LargeVehicles", "Landmarks"]
    feats = morning_cols + static_cols + cat_cols
    for c in cat_cols:
        tr[c] = tr[c].astype(str).fillna("Missing"); te[c] = te[c].astype(str).fillna("Missing")
    for c in morning_cols + static_cols:
        tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(-1.0)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(-1.0)
    return tr, te, feats, cat_cols


def main():
    tr, te, feats, cat_cols = build()
    cat_idx = [feats.index(c) for c in cat_cols]
    y = tr["demand"].to_numpy(); X = tr[feats]; groups = tr["geohash"].to_numpy()
    print(f"train={len(tr)} test={len(te)} feats={len(feats)}", flush=True)

    gkf = GroupKFold(n_splits=5)
    oof_cat = np.zeros(len(tr))
    for fold, (tri, vai) in enumerate(gkf.split(X, y, groups)):
        t0 = time.time()
        mc = CatBoostRegressor(iterations=2000, learning_rate=0.035, depth=9, l2_leaf_reg=6,
                               random_seed=2026 + fold, verbose=0, allow_writing_files=False)
        mc.fit(Pool(X.iloc[tri], y[tri], cat_features=cat_idx))
        oof_cat[vai] = np.clip(mc.predict(X.iloc[vai]), 0, 1)
        print(f"  fold {fold}: cat={r2_score(y[vai],oof_cat[vai]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  HONEST cat-v2 R2={r2_score(y,oof_cat):.5f}  (v1 was 0.80334)", flush=True)

    mc = CatBoostRegressor(iterations=2400, learning_rate=0.035, depth=9, l2_leaf_reg=6,
                           random_seed=2026, verbose=0, allow_writing_files=False)
    mc.fit(Pool(X, y, cat_features=cat_idx))
    cat_t = np.clip(mc.predict(te[feats]), 0, 1)

    best_df = pd.read_csv(OUT_DIR / "anchor_w5.csv"); b = best_df["demand"].to_numpy()
    te_sorted = te.sort_values("Index")
    m2d = pd.Series(cat_t, index=te.index).loc[te_sorted.index].to_numpy()
    assert (te_sorted["Index"].to_numpy() == best_df["Index"].to_numpy()).all()

    # also blend v1+v2 m2d for a smoother signal
    v1 = pd.read_csv(OUT_DIR / "m2d_pure.csv")["demand"].to_numpy()
    m2d_avg = 0.5 * m2d + 0.5 * v1

    def write(name, arr):
        pd.DataFrame({"Index": best_df["Index"].astype(int), "demand": np.clip(arr, 0, 1)}).to_csv(OUT_DIR / name, index=False)
        print(f"  {name:<30} chg_vs_best%={np.mean(np.abs(arr-b)>0.005)*100:.2f}", flush=True)

    write("m2dv2_pure.csv", m2d)
    for w in [0.06, 0.08, 0.10, 0.12]:
        write(f"m2dv2_mix_w{int(w*100)}.csv", (1 - w) * b + w * m2d)
    for w in [0.08, 0.10, 0.12]:
        write(f"m2dboth_mix_w{int(w*100)}.csv", (1 - w) * b + w * m2d_avg)
    print("done", flush=True)


if __name__ == "__main__":
    main()
