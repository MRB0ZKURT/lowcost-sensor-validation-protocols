"""Stage 7 - permutation importance in and out of regime (Sect. 4.7, Fig. 3).

Output
    results/permutation_importance.csv
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from common import (FEATURES, RANDOM_STATE, RESULTS, TARGETS, TRAIN_END_BLOCK,
                    design, load_primary)

N_REPEATS = 10


def main() -> None:
    df = load_primary()
    rows = []
    t0 = time.time()
    for target in TARGETS:
        d, X, y, blk = design(df, target)
        tr = blk <= TRAIN_END_BLOCK
        te = blk > TRAIN_END_BLOCK
        rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                   n_jobs=1).fit(X[tr], y[tr])
        for split_name, mask in (("train period (in-regime)", tr),
                                 ("deployment period (out-of-regime)", te)):
            pi = permutation_importance(rf, X[mask], y[mask], n_repeats=N_REPEATS,
                                        random_state=RANDOM_STATE, n_jobs=1,
                                        scoring="neg_root_mean_squared_error")
            for f, mu, sd in zip(FEATURES, pi.importances_mean, pi.importances_std):
                rows.append(dict(target=target, split=split_name, feature=f,
                                 importance=float(mu), sd=float(sd)))
        print(f"[s07] {target} done {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "permutation_importance.csv", index=False)
    print("[s07] wrote permutation_importance.csv")


if __name__ == "__main__":
    main()
