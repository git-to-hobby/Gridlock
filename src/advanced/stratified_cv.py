"""Stratified K-Fold splits that preserve extreme demand bins."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def demand_bins(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Quantile bins with top bin forced to capture y == 1.0 extremes."""
    y = np.asarray(y, dtype=float)
    # Custom edges: extra granularity at top tail
    edges = [0.0, 0.05, 0.15, 0.35, 0.65, 0.90, 1.0001]
    bins = np.digitize(y, edges, right=False) - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    # Force exact 1.0 into top bin
    bins[y >= 0.99] = len(edges) - 2
    return bins.astype(int)


def stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) with stratified target bins."""
    bins = demand_bins(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), bins))


def assign_fold_ids(y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """Per-row fold id for OOF target encoding aligned with stratified CV."""
    fold_id = np.full(len(y), -1, dtype=int)
    for f, (_, va) in enumerate(stratified_kfold_indices(y, n_splits, random_state)):
        fold_id[va] = f
    return fold_id


def assign_fold_ids_geohash(
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Fold ids from GroupKFold on geohash — matches hierarchical TE splits."""
    from sklearn.model_selection import GroupKFold

    fold_id = np.full(len(groups), -1, dtype=int)
    gkf = GroupKFold(n_splits=n_splits)
    for f, (_, va) in enumerate(gkf.split(np.zeros(len(groups)), np.zeros(len(groups)), groups)):
        fold_id[va] = f
    return fold_id
