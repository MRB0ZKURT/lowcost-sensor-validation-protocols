"""Stage 6 - unsupervised drift corrections under the deployment protocol (Sect. 4.5).

Three corrections that need no reference data are compared against the raw
signals: a causal 30-day rolling-median baseline, a log-log transform, and an
elapsed-time predictor.

Output
    results/drift_corrections.csv
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from common import (METEO, REPRESENTATIVE, RESULTS, SENSORS, TARGETS,
                    TRAIN_END_BLOCK, build, load_primary)

VARIANTS = {
    "none (raw sensor signals)": dict(feats=SENSORS + METEO, logx=False, logy=False),
    "log-log transform": dict(feats=SENSORS + METEO, logx=True, logy=True),
    "rolling-baseline normalisation": dict(feats=[s + "_norm" for s in SENSORS] + METEO,
                                           logx=False, logy=False),
    "elapsed-time feature": dict(feats=SENSORS + METEO + ["elapsed_h"],
                                 logx=False, logy=False),
}


def augment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    df["elapsed_h"] = (df.datetime - df.datetime.min()).dt.total_seconds() / 3600.0
    # Strictly causal baseline: the window is closed on the left, so the current
    # hour never contributes to its own baseline.
    base = df.set_index("datetime")[SENSORS].rolling(
        "30D", min_periods=168, closed="left").median()
    for s in SENSORS:
        df[s + "_norm"] = df[s].to_numpy() / base[s].to_numpy()
    return df


def main() -> None:
    df = augment(load_primary())
    rows = []
    t0 = time.time()
    for target in TARGETS:
        for vname, v in VARIANTS.items():
            cols = v["feats"]
            d = df[["datetime", "block"] + cols + [target]].dropna().reset_index(drop=True)
            # copy=True: the log transform below writes in place, and pandas
            # returns a read-only view under copy-on-write.
            X = d[cols].to_numpy(dtype=float, copy=True)
            y = d[target].to_numpy(float)
            if v["logx"]:
                for j, c in enumerate(cols):
                    if c.startswith("PT08"):
                        X[:, j] = np.log(X[:, j])
            blk = d["block"].to_numpy()
            tr = np.where(blk <= TRAIN_END_BLOCK)[0]
            te = np.where(blk > TRAIN_END_BLOCK)[0]
            for name in REPRESENTATIVE:
                m = build(name).fit(X[tr], np.log(y[tr]) if v["logy"] else y[tr])
                p = m.predict(X[te])
                if v["logy"]:
                    p = np.exp(p)
                rows.append(dict(target=target, variant=vname, model=name,
                                 n_train=len(tr), n_test=len(te),
                                 rmse=float(np.sqrt(mean_squared_error(y[te], p))),
                                 r2=float(r2_score(y[te], p))))
        print(f"[s06] {target} done {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "drift_corrections.csv", index=False)
    print("[s06] wrote drift_corrections.csv")


if __name__ == "__main__":
    main()
