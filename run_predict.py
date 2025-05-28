# rebuild final csv from saved preds

import numpy as np
import pandas as pd
from pathlib import Path

root = Path(__file__).parent
o = root / "output"

a = pd.read_csv(o / "push_triple_w145.csv").sort_values("Index")
b = pd.read_csv(o / "v3adv_lgbm.csv").sort_values("Index")
df = a.merge(b, on="Index", suffixes=("_x", "_y"))

if (o / "v3adv_w145lgb_0983.csv").exists():
    base = pd.read_csv(o / "v3adv_w145lgb_0983.csv").sort_values("Index").demand.values
else:
    base = 0.983 * df.demand_x + 0.017 * df.demand_y

lg = df.demand_y.values
top = lg >= np.quantile(lg, 0.996)
pred = base.copy()
pred[top] = np.maximum(pred[top], 0.97 * lg[top])
pred = np.clip(pred, 0, 1)

pd.DataFrame({"Index": df.Index.astype(int), "demand": pred}).to_csv(
    o / "submission.csv", index=False, float_format="%.16g", lineterminator="\n"
)
print("ok mean=", pred.mean())
