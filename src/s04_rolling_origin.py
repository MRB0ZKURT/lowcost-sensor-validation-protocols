"""Stage 4 - rolling-origin design separating horizon from season (Sect. 4.3, Fig. 4).

For each training origin s the model is fitted on blocks s..s+5 (six months) and
evaluated on block s+5+h for horizons h = 1..4 months, so that each horizon is
observed in several seasons and each season at several horizons.

Output
    results/rolling_origin.csv
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from common import (BLOCK_MONTH, REPRESENTATIVE, RESULTS, TARGETS, build,
                    design, load_primary)

ORIGINS = range(0, 7)
HORIZONS = range(1, 5)
WINDOW = 6  # months of calibration data


def main() -> None:
    df = load_primary()
    rows = []
    t0 = time.time()
    for target in TARGETS:
        d, X, y, blk = design(df, target)
        for s in ORIGINS:
            tr = np.where((blk >= s) & (blk <= s + WINDOW - 1))[0]
            if len(tr) < 500:
                continue
            for name in REPRESENTATIVE:
                m = build(name).fit(X[tr], y[tr])
                for h in HORIZONS:
                    b = s + WINDOW - 1 + h
                    if b > max(BLOCK_MONTH):
                        continue
                    te = np.where(blk == b)[0]
                    if len(te) < 50:
                        continue
                    rows.append(dict(
                        target=target, model=name, origin=s, horizon=h, test_block=b,
                        test_month=BLOCK_MONTH[b], n_train=int(len(tr)),
                        n_test=int(len(te)),
                        rmse=float(np.sqrt(mean_squared_error(y[te], m.predict(X[te]))))))
        print(f"[s04] {target} done {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "rolling_origin.csv", index=False)
    print(f"[s04] wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
