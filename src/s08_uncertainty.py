"""Stage 8 - relative expanded uncertainty at the limit value (Sect. 4.4, Fig. 5).

Out-of-sample predictions are regenerated for the three representative models
under all three protocols, an orthogonal regression is fitted between prediction
and reference, and the relative expanded uncertainty is evaluated at the EU
limit value. The uncertainty of the reference method is set to zero, which makes
the reported values conservative upper bounds.

Outputs
    results/predictions_all_protocols.parquet
    results/table_reu.csv
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import (LIMIT_VALUES, RANDOM_STATE, REPRESENTATIVE, RESULTS, TARGETS,
                    TRAIN_END_BLOCK, build, design, load_primary,
                    orthogonal_regression)


def predictions(df) -> pd.DataFrame:
    frames = []
    t0 = time.time()
    for target in TARGETS:
        d, X, y, blk = design(df, target)
        for name in REPRESENTATIVE:
            p = np.full(len(y), np.nan)
            for tr, te in KFold(10, shuffle=True, random_state=RANDOM_STATE).split(X):
                p[te] = build(name).fit(X[tr], y[tr]).predict(X[te])
            frames.append(pd.DataFrame(dict(target=target, model=name,
                                            protocol="Random 10-fold CV", y=y, pred=p)))

            p = np.full(len(y), np.nan)
            for k in range(2, 12):
                tr, te = np.where(blk <= k)[0], np.where(blk == k + 1)[0]
                if len(te) < 50:
                    continue
                p[te] = build(name).fit(X[tr], y[tr]).predict(X[te])
            ok = ~np.isnan(p)
            frames.append(pd.DataFrame(dict(target=target, model=name,
                                            protocol="Forward-chaining CV",
                                            y=y[ok], pred=p[ok])))

            tr = np.where(blk <= TRAIN_END_BLOCK)[0]
            te = np.where(blk > TRAIN_END_BLOCK)[0]
            m = build(name).fit(X[tr], y[tr])
            frames.append(pd.DataFrame(dict(target=target, model=name,
                                            protocol="Deployment split",
                                            y=y[te], pred=m.predict(X[te]))))
        print(f"[s08] {target} predictions done {time.time() - t0:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    P = predictions(load_primary())
    P.to_parquet(RESULTS / "predictions_all_protocols.parquet", index=False)

    rows = []
    for (target, model, protocol), g in P.groupby(["target", "model", "protocol"]):
        x, yv = g.y.to_numpy(), g.pred.to_numpy()
        n = len(x)
        a, b = orthogonal_regression(x, yv)
        rss = float(((yv - a - b * x) ** 2).sum())
        rec = dict(target=target, model=model, protocol=protocol, n=n, slope=b,
                   intercept=a, rmse=float(np.sqrt(((yv - x) ** 2).mean())),
                   mbe=float((yv - x).mean()))
        lv = LIMIT_VALUES[target]
        if lv is not None:
            # u_ref is taken as zero, so this is an upper bound on the REU.
            u = np.sqrt(rss / (n - 2) + (a + (b - 1) * lv) ** 2)
            rec["REU_pct_at_LV"] = float(200 * u / lv)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(RESULTS / "table_reu.csv", index=False)
    print("[s08] wrote table_reu.csv")


if __name__ == "__main__":
    main()
