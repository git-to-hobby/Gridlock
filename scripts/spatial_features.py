#!/usr/bin/env python3
"""Geohash → lat/lon decoder + neighbour-graph features.

Standard 6-char geohashes (base32). The dataset's prefixes (`qp02…`, `qp08…`)
sit in a small region, so decoded lat/lon become very informative tabular
features and the K-NN demand aggregates capture local spatial smoothness.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
DECODE = {c: i for i, c in enumerate(BASE32)}


def decode_geohash(gh: str) -> Tuple[float, float, float, float]:
    """Decode geohash → (lat, lon, lat_err, lon_err)."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for ch in gh:
        cd = DECODE[ch]
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                if cd & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if cd & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (
        (lat_lo + lat_hi) / 2,
        (lon_lo + lon_hi) / 2,
        (lat_hi - lat_lo) / 2,
        (lon_hi - lon_lo) / 2,
    )


def decode_series(geohashes: pd.Series) -> pd.DataFrame:
    uniq = geohashes.dropna().unique()
    rows = {g: decode_geohash(g) for g in uniq}
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["lat", "lon", "lat_err", "lon_err"])
    out["lat_lon_norm"] = np.sqrt(out["lat"] ** 2 + out["lon"] ** 2)
    return out


def add_spatial_features(train_frame: pd.DataFrame, test_frame: pd.DataFrame,
                          neighbour_k: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add lat/lon, distance to centroid, and K-NN demand aggregates.

    Leakage-safe rule: K-NN demand features use **day-48** demand only (already
    available in both train and test rows). The day-48 demand profile is
    pre-computed in the forecast frame as `d48_gh_mean` and is identical for a
    given geohash regardless of which row it appears in, so this is safe.
    """
    tr = train_frame.copy()
    te = test_frame.copy()

    all_gh = pd.concat([tr["geohash"], te["geohash"]]).dropna().unique()
    decoded = pd.DataFrame.from_dict(
        {g: decode_geohash(g) for g in all_gh},
        orient="index", columns=["lat", "lon", "lat_err", "lon_err"],
    )
    decoded["lat_lon_norm"] = np.sqrt(decoded["lat"] ** 2 + decoded["lon"] ** 2)

    for frame in (tr, te):
        frame["gh_lat"] = frame["geohash"].map(decoded["lat"])
        frame["gh_lon"] = frame["geohash"].map(decoded["lon"])
        frame["gh_lat_err"] = frame["geohash"].map(decoded["lat_err"])
        frame["gh_lon_err"] = frame["geohash"].map(decoded["lon_err"])
        frame["gh_norm"] = frame["geohash"].map(decoded["lat_lon_norm"])

    # Centroid-relative features
    centroid_lat = decoded["lat"].mean()
    centroid_lon = decoded["lon"].mean()
    for frame in (tr, te):
        frame["gh_dlat"] = frame["gh_lat"] - centroid_lat
        frame["gh_dlon"] = frame["gh_lon"] - centroid_lon
        frame["gh_dist_centroid"] = np.sqrt(frame["gh_dlat"] ** 2 + frame["gh_dlon"] ** 2)

    # K-NN demand aggregates using **day-48 mean demand per geohash** as the
    # value at each cell. This is identical to the existing d48_gh_mean column,
    # so it does not introduce new leakage.
    if "d48_gh_mean" in tr.columns:
        gh_to_d48 = (
            pd.concat([tr[["geohash", "d48_gh_mean"]], te[["geohash", "d48_gh_mean"]]])
            .dropna(subset=["geohash"])
            .groupby("geohash")["d48_gh_mean"].mean()
        )
    else:
        gh_to_d48 = pd.Series(dtype=float)

    coords = decoded[["lat", "lon"]].to_numpy()
    gh_index = decoded.index.to_numpy()

    if len(coords) >= neighbour_k + 1:
        tree = BallTree(np.radians(coords), metric="haversine")
        # k+1 because the first match is the cell itself
        dists, idxs = tree.query(np.radians(coords), k=neighbour_k + 1)
        global_mean = float(gh_to_d48.mean()) if not gh_to_d48.empty else 0.094

        knn_mean, knn_std, knn_w, knn_count = [], [], [], []
        for row_dists, row_idxs in zip(dists, idxs):
            neighbour_gh = gh_index[row_idxs[1:]]
            vals = np.array([gh_to_d48.get(g, global_mean) for g in neighbour_gh])
            knn_mean.append(float(vals.mean()))
            knn_std.append(float(vals.std()))
            # Inverse-distance-weighted average (avoid div-by-zero)
            w = 1.0 / (row_dists[1:] + 1e-6)
            knn_w.append(float((vals * w).sum() / w.sum()))
            knn_count.append(int((vals > global_mean).sum()))

        knn = pd.DataFrame({
            "knn_d48_mean": knn_mean,
            "knn_d48_std": knn_std,
            "knn_d48_wmean": knn_w,
            "knn_high_neighbours": knn_count,
        }, index=gh_index)

        for col in knn.columns:
            tr[col] = tr["geohash"].map(knn[col])
            te[col] = te["geohash"].map(knn[col])
            tr[col] = tr[col].fillna(tr[col].median())
            te[col] = te[col].fillna(te[col].median())

    return tr, te
