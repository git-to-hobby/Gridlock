#!/usr/bin/env python3
"""Morning -> daytime forecaster (the lever the main model structurally cannot learn).

The main model cannot learn "given day-49 morning, predict day-49 daytime" because no
training row pairs a day-49 morning with a daytime target. But DAY 48 has the full
morning->daytime relationship labeled. So:

  TRAIN: features = (day-48 morning slots 0-8 trajectory per geohash) + static + target slot
         target   = day-48 demand at that daytime slot (9-55)
  APPLY: features = (day-49 morning slots 0-8 trajectory)            + static + target slot
         predict  = day-49 daytime demand (= TEST)

We do NOT use day-48 same-slot daytime value as a feature (that would be the label on
day 48). The model instead learns how the morning shape maps to the rest of the day.

Honest validation: GroupKFold by geohash on day-48 daytime (train geohashes -> predict
held-out geohashes), so we measure true generalization of the morning->daytime mapping.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DATA_DIR = REPO_ROOT / "dataset"
OUT_DIR = REPO_ROOT / "output"

MORNING_SLOTS = list(range(0, 9))  # 0..8


def add_time(df):
    out = df.copy()
    p = out["timestamp"].astype(str).str.split(":", expand=True).astype(int)
    out["hour"] = p[0]
    out["minute"] = p[1]
    out["slot"] = p[0] * 4 + p[1] // 15
    out["slot_sin"] = np.sin(2 * np.pi * out["slot"] / 96)
    out["slot_cos"] = np.cos(2 * np.pi * out["slot"] / 96)
    for pr in (3, 4):
        out[f"gh_p{pr}"] = out["geohash"].str[:pr]
    return out


def morning_profile(day_df: pd.DataFrame) -> pd.DataFrame:
    """Per-geohash morning trajectory features from slots 0-8."""
    m = day_df[day_df["slot"].isin(MORNING_SLOTS)]
    piv = m.pivot_table(index="geohash", columns="slot", values="demand")
    piv = piv.reindex(columns=MORNING_SLOTS)
    feats = pd.DataFrame(index=piv.index)
    for s in MORNING_SLOTS:
        feats[f"m{s}"] = piv[s]
    feats["m_mean"] = piv.mean(axis=1)
    feats["m_std"] = piv.std(axis=1)
    feats["m_max"] = piv.max(axis=1)
    feats["m_min"] = piv.min(axis=1)
    feats["m_last"] = piv[8]
    feats["m_first"] = piv[0]
    # linear trend across morning slots
    x = np.array(MORNING_SLOTS, dtype=float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    vals = piv.to_numpy()
    slope = np.nansum((vals - np.nanmean(vals, axis=1, keepdims=True)) * xc, axis=1) / denom
    feats["m_trend"] = slope
    feats["m_range"] = feats["m_max"] - feats["m_min"]
    return feats.reset_index()


def build():
    raw = add_time(pd.read_csv(DATA_DIR / "train.csv"))
    test = add_time(pd.read_csv(DATA_DIR / "test.csv"))
    for f in (raw, test):
        for c in ("RoadType", "Weather", "LargeVehicles", "Landmarks"):
            f[c] = f[c].fillna("Missing")
        f["Temperature"] = f["Temperature"].fillna(raw["Temperature"].median())

    d48 = raw[raw["day"] == 48]
    d49 = raw[raw["day"] == 49]

    prof48 = morning_profile(d48)
    prof49 = morning_profile(d49)

    # TRAIN rows: day-48 daytime (slots 9-55) joined with day-48 morning profile
    tr = d48[d48["slot"].between(9, 55)].merge(prof48, on="geohash", how="left")
    # APPLY rows: test (day-49 daytime) joined with day-49 morning profile
    te = test.merge(prof49, on="geohash", how="left")

    morning_cols = [f"m{s}" for s in MORNING_SLOTS] + [
        "m_mean", "m_std", "m_max", "m_min", "m_last", "m_first", "m_trend", "m_range"
    ]
    static_cols = ["slot", "hour", "slot_sin", "slot_cos", "Temperature", "NumberofLanes"]
    cat_cols = ["geohash", "gh_p4", "RoadType", "Weather", "LargeVehicles", "Landmarks"]
    feature_cols = morning_cols + static_cols + cat_cols

    for c in cat_cols:
        tr[c] = tr[c].astype(str).fillna("Missing")
        te[c] = te[c].astype(str).fillna("Missing")
    for c in morning_cols + static_cols:
        tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(-1.0)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(-1.0)
    return tr, te, feature_cols, cat_cols


def main():
    tr, te, feature_cols, cat_cols = build()
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    y = tr["demand"].to_numpy()
    X = tr[feature_cols]
    groups = tr["geohash"].to_numpy()
    print(f"train rows={len(tr)} (day-48 daytime)  test rows={len(te)}", flush=True)

    # Honest geohash-split validation
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(tr))
    for fold, (tri, vai) in enumerate(gkf.split(X, y, groups)):
        t0 = time.time()
        m = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=8, l2_leaf_reg=6,
                              random_seed=2026 + fold, verbose=0, allow_writing_files=False)
        m.fit(Pool(X.iloc[tri], y[tri], cat_features=cat_idx))
        oof[vai] = np.clip(m.predict(X.iloc[vai]), 0, 1)
        print(f"  fold {fold}: groupR2={r2_score(y[vai], oof[vai]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    honest = r2_score(y, oof)
    print(f"\n  HONEST geohash-split R2 (morning->daytime on day48) = {honest:.5f}", flush=True)

    # naive ref: day-48 same-slot can't be used (it's the label). Compare to morning mean baseline.
    ref = r2_score(y, np.clip(tr["m_mean"].to_numpy(), 0, 1))
    print(f"  [ref] morning-mean alone R2 = {ref:.5f}", flush=True)

    # Train final on all day-48 daytime, predict test (day-49 daytime)
    print("\n  Training final on all day-48 daytime...", flush=True)
    mfull = CatBoostRegressor(iterations=1800, learning_rate=0.04, depth=8, l2_leaf_reg=6,
                              random_seed=2026, verbose=0, allow_writing_files=False)
    mfull.fit(Pool(X, y, cat_features=cat_idx))
    test_pred = np.clip(mfull.predict(te[feature_cols]), 0, 1)

    best_df = pd.read_csv(OUT_DIR / "anchor_w5.csv")
    best = best_df["demand"].to_numpy()
    te2 = te.sort_values("Index").reset_index(drop=True)
    assert (te2["Index"].to_numpy() == best_df["Index"].to_numpy()).all()
    order = te.sort_values("Index").index
    m2d = pd.Series(test_pred, index=te.index).loc[order].to_numpy()

    print(f"\n  morning->daytime test pred: mean={m2d.mean():.4f} std={m2d.std():.4f}", flush=True)
    print(f"  corr with anchor_w5: {np.corrcoef(best, m2d)[0,1]:.4f}", flush=True)

    def write(name, arr):
        pd.DataFrame({"Index": best_df["Index"].astype(int), "demand": np.clip(arr, 0, 1)}).to_csv(OUT_DIR / name, index=False)
        d = np.abs(arr - best)
        print(f"  {name:<32} mean={np.clip(arr,0,1).mean():.5f} chg_vs_best%={np.mean(d>0.005)*100:.2f}", flush=True)

    write("m2d_pure.csv", m2d)
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        write(f"m2d_mix_w{int(w*100)}.csv", (1 - w) * best + w * m2d)


if __name__ == "__main__":
    main()
