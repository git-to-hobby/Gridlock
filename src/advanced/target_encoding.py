"""Leakage-safe K-Fold target encoding with smoothing."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _smooth_mean(counts: np.ndarray, sums: np.ndarray, global_mean: float, m: float) -> np.ndarray:
    return (sums + m * global_mean) / (counts + m)


def kfold_target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Iterable[str],
    target: str = "demand",
    n_splits: int = 5,
    smoothing: float = 20.0,
    random_state: int = 42,
    fold_id: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Out-of-fold target encoding for train; full-train maps for test.

    If ``fold_id`` is provided (same length as train), uses those fold ids
    instead of random KFold — useful with stratified splits.

    Returns train, test with new ``{col}_te`` columns and list of new col names.
    """
    from sklearn.model_selection import KFold

    train = train.copy()
    test = test.copy()
    y = train[target].to_numpy()
    global_mean = float(np.mean(y))
    new_cols: list[str] = []

    if fold_id is None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_iter = list(kf.split(train))
    else:
        fold_iter = []
        for f in range(n_splits):
            va = np.where(fold_id == f)[0]
            tr = np.where(fold_id != f)[0]
            fold_iter.append((tr, va))

    for col in cols:
        te_col = f"{col}_te"
        new_cols.append(te_col)
        train[te_col] = np.nan

        for tr_idx, va_idx in fold_iter:
            tr_part = train.iloc[tr_idx]
            stats = tr_part.groupby(col)[target].agg(["sum", "count"])
            enc = _smooth_mean(
                stats["count"].to_numpy(),
                stats["sum"].to_numpy(),
                global_mean,
                smoothing,
            )
            mapping = pd.Series(enc, index=stats.index)
            train.loc[train.index[va_idx], te_col] = (
                train.iloc[va_idx][col].map(mapping).astype(float)
            )

        train[te_col] = train[te_col].fillna(global_mean)

        full_stats = train.groupby(col)[target].agg(["sum", "count"])
        enc_full = _smooth_mean(
            full_stats["count"].to_numpy(),
            full_stats["sum"].to_numpy(),
            global_mean,
            smoothing,
        )
        mapping_full = pd.Series(enc_full, index=full_stats.index)
        test[te_col] = test[col].map(mapping_full).astype(float).fillna(global_mean)

    return train, test, new_cols


def multi_col_target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    interaction_cols: list[list[str]],
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Target-encode composite keys like geohash×slot."""
    train = train.copy()
    test = test.copy()
    all_new: list[str] = []
    for cols in interaction_cols:
        key = "_".join(cols)
        train[key] = train[cols[0]].astype(str)
        test[key] = test[cols[0]].astype(str)
        for c in cols[1:]:
            train[key] = train[key] + "|" + train[c].astype(str)
            test[key] = test[key] + "|" + test[c].astype(str)
        train, test, new_cols = kfold_target_encode(train, test, [key], **kwargs)
        all_new.extend(new_cols)
    return train, test, all_new
