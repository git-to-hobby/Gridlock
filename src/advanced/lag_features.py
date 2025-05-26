"""Lag and rolling features along geohash timelines (leakage-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_rolling_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_col: str = "geohash",
    time_cols: tuple[str, str] = ("day", "slot"),
    target: str = "demand",
    lags: tuple[int, ...] = (1, 2),
    windows: tuple[int, ...] = (3, 6),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build lag / rolling stats using only past observations per geohash.

    Test rows use train history only (test ``demand`` is NaN during build).
    Original row order is preserved in returned frames.
    """
    train = train.copy()
    test = test.copy()
    train["_row_id"] = np.arange(len(train))
    test["_row_id"] = np.arange(len(test))
    test[target] = np.nan
    test["_is_test"] = 1
    train["_is_test"] = 0

    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.sort_values([group_col, time_cols[0], time_cols[1]]).reset_index(drop=True)

    new_cols: list[str] = []
    grp = combined.groupby(group_col, sort=False)[target]

    for lag in lags:
        col = f"lag_{lag}"
        combined[col] = grp.shift(lag)
        new_cols.append(col)

    shifted = grp.shift(1)
    for w in windows:
        for suffix, fn in [("mean", "mean"), ("max", "max"), ("std", "std")]:
            col = f"roll{w}_{suffix}"
            combined[col] = shifted.rolling(window=w, min_periods=1).agg(fn).reset_index(level=0, drop=True)
            new_cols.append(col)

    fallback = combined["d48_same_slot"] if "d48_same_slot" in combined.columns else float(train[target].mean())
    if isinstance(fallback, pd.Series):
        for col in new_cols:
            combined[col] = combined[col].fillna(fallback).fillna(float(train[target].mean()))
    else:
        for col in new_cols:
            combined[col] = combined[col].fillna(fallback)

    tr = (
        combined[combined["_is_test"] == 0]
        .sort_values("_row_id")
        .drop(columns=["_is_test", "_row_id"])
        .reset_index(drop=True)
    )
    te = (
        combined[combined["_is_test"] == 1]
        .sort_values("_row_id")
        .drop(columns=["_is_test", "_row_id", target], errors="ignore")
        .reset_index(drop=True)
    )
    return tr, te, new_cols
