#!/usr/bin/env python3
"""Apply tail max-correction on top of LB-best submission.

Winning rule (LB 91.93890): on top ~0.4% LGBM scores,
  pred = max(base, coef * lgbm_asymmetric)

Usage:
  python apply_mega_jump.py                    # default q=0.996, coef=0.97
  python apply_mega_jump.py --base jump_max_q996.csv --q 0.996 --coef 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"


def apply_max_jump(
    base_path: Path,
    lgbm_path: Path,
    *,
    quantile: float = 0.996,
    coef: float = 0.97,
    gap_min: float | None = None,
    clip_one: float | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    base = pd.read_csv(base_path).sort_values("Index")
    lgbm = pd.read_csv(lgbm_path).sort_values("Index")
    m = base.merge(lgbm, on="Index", suffixes=("_base", "_lgb"))
    b = m["demand_base"].to_numpy()
    l = m["demand_lgb"].to_numpy()
    out = b.copy()

    if gap_min is not None:
        mask = (l - b) >= gap_min
    else:
        mask = l >= np.quantile(l, quantile)

    out[mask] = np.maximum(b[mask], coef * l[mask])
    if clip_one is not None:
        hot = l >= clip_one
        out[hot] = np.maximum(out[hot], 1.0)

    out = np.clip(out, 0, 1)
    result = pd.DataFrame({"Index": m["Index"].astype(int), "demand": out})
    if out_path:
        result.to_csv(out_path, index=False, float_format="%.16g", lineterminator="\n")
        chg = int((np.abs(out - b) > 1e-6).sum())
        print(f"Wrote {out_path.name}: changed={chg}  >=0.99={(out>=0.99).sum()}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="v3adv_w145lgb_0983.csv")
    p.add_argument("--lgbm", default="v3adv_lgbm.csv")
    p.add_argument("--q", type=float, default=0.996)
    p.add_argument("--coef", type=float, default=0.97)
    p.add_argument("--gap", type=float, default=None, help="Use gap threshold instead of quantile")
    p.add_argument("--clip-one", type=float, default=None, help="Force 1.0 where lgbm >= this")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_name = args.out or f"jump_custom_q{int(args.q*1000)}_c{int(args.coef*1000)}.csv"
    apply_max_jump(
        OUTPUT_DIR / args.base,
        OUTPUT_DIR / args.lgbm,
        quantile=args.q,
        coef=args.coef,
        gap_min=args.gap,
        clip_one=args.clip_one,
        out_path=OUTPUT_DIR / out_name,
    )


if __name__ == "__main__":
    main()
