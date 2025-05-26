"""Asymmetric loss utilities — penalize under-prediction on high-demand rows."""

from __future__ import annotations

import numpy as np


def asymmetric_sample_weights(
    y: np.ndarray,
    *,
    high_threshold: float = 0.85,
    extreme_threshold: float = 0.98,
    high_boost: float = 2.5,
    extreme_boost: float = 5.0,
) -> np.ndarray:
    """Static sample weights for CatBoost / weighted training.

    Rows with demand >= high_threshold get higher weight so the model
    pays more attention to the upper tail during training.
    """
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y), dtype=float)
    w = np.where(y >= high_threshold, w * high_boost, w)
    w = np.where(y >= extreme_threshold, w * extreme_boost, w)
    return w


def lgbm_asymmetric_objective(
    alpha: float = 3.0,
    high_y: float = 0.80,
):
    """LightGBM custom objective for sklearn API: ``(y_true, y_pred) -> (grad, hess)``.

    Under-prediction on high targets gets ``alpha`` times larger gradient.
    """

    def _grad_hess(y: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float).reshape(-1)
        preds = np.asarray(preds, dtype=float).reshape(-1)
        residual = preds - y
        under = (y >= high_y) & (residual < 0)
        weight = np.where(under, alpha, 1.0)
        grad = 2.0 * residual * weight
        hess = 2.0 * weight
        return grad, hess

    def objective(y_true, y_pred):
        # sklearn LGBMRegressor passes (labels, preds)
        return _grad_hess(y_true, y_pred)

    def eval_metric(y_pred, y_true):
        # sklearn eval callback: (preds, labels)
        y = np.asarray(y_true, dtype=float).reshape(-1)
        preds = np.asarray(y_pred, dtype=float).reshape(-1)
        residual = preds - y
        under = (y >= high_y) & (residual < 0)
        weight = np.where(under, alpha, 1.0)
        loss = float(np.mean(weight * residual ** 2))
        return "asymmetric_mse", loss, False

    return objective, eval_metric


def catboost_asymmetric_metric():
    """CatBoost custom metric for eval: asymmetric MSE on validation."""

    def metric(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        residual = y_pred - y_true
        under = (y_true >= 0.80) & (residual < 0)
        weight = np.where(under, 3.0, 1.0)
        return float(np.mean(weight * residual ** 2)), 1.0

    return metric
