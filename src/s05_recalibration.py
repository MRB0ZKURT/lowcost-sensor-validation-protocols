"""Stage 5 - sliding-window recalibration (Sect. 4.5, Fig. 2).

For every deployment month the model is refitted on the W hours immediately
preceding that month, and separately on the entire accumulated history. The
comparison baseline is the never-updated deployment model scored on the same
monthly basis, which is read back from stage 2.

Outputs
    results/recalibration_results.csv
    results/table_recalibration.csv
    results/table_drift_by_month.csv
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from common import (REPRESENTATIVE, RESULTS, TARGETS, TRAIN_END_BLOCK, build,
                    design, load_primary)

WINDOWS = [168, 336, 720, 2160, 4320, None]  # 1 wk, 2 wk, 1 mo, 3 mo, 6 mo, all


def sliding_window(df) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for target in TARGETS:
        d, X, y, blk = design(df, target)
        for m_block in range(TRAIN_END_BLOCK + 1, 13):
            te = np.where(blk == m_block)[0]
            if len(te) < 50:
                continue
            t_start = d.loc[te, "datetime"].min()
            for W in WINDOWS:
                if W is None:
                    tr = np.where(d.datetime < t_start)[0]
                else:
                    tr = np.where((d.datetime < t_start) &
                                  (d.datetime >= t_start - pd.Timedelta(hours=W)))[0]
                if len(tr) < 100:
                    continue
                for name in REPRESENTATIVE:
                    mdl = build(name).fit(X[tr], y[tr])
                    p = mdl.predict(X[te])
                    rows.append(dict(target=target, model=name,
                                     window_h=(-1 if W is None else W),
                                     test_block=m_block, n_train=int(len(tr)),
                                     rmse=float(np.sqrt(mean_squared_error(y[te], p))),
                                     r2=float(r2_score(y[te], p))))
        print(f"[s05] {target} done {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def never_updated_baseline() -> pd.DataFrame:
    """Per-month RMSE of the fixed deployment model, from stage 2 predictions."""
    pred = pd.read_parquet(RESULTS / "deployment_predictions.parquet")
    pred = pred[pred.model.isin(REPRESENTATIVE)]
    out = (pred.groupby(["target", "model", "block"])
           .apply(lambda g: np.sqrt(mean_squared_error(g.y, g.pred)),
                  include_groups=False)
           .rename("rmse").reset_index())
    return out


def main() -> None:
    df = load_primary()
    rc = sliding_window(df)
    rc.to_csv(RESULTS / "recalibration_results.csv", index=False)

    base = never_updated_baseline()
    base.to_csv(RESULTS / "table_drift_by_month.csv", index=False)

    piv = rc.pivot_table(index=["target", "model"], columns="window_h",
                         values="rmse", aggfunc="mean")
    piv.insert(0, "never updated", base.groupby(["target", "model"]).rmse.mean())
    piv.columns = ["never updated"] + [("all past" if c == -1 else f"{int(c)} h")
                                       for c in piv.columns[1:]]
    piv.round(3).to_csv(RESULTS / "table_recalibration.csv")
    print("[s05] wrote recalibration tables")


if __name__ == "__main__":
    main()
