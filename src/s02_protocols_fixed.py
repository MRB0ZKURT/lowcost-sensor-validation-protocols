"""Stage 2 - protocol comparison with fixed hyperparameters (Sect. 4.1, Fig. 1).

The random forest uses 500 trees here; the experiments that refit many times
(stages 4-7) use 300. Both settings are recorded in the manuscript.

Outputs
    results/model_results_raw.csv            441 rows
    results/deployment_predictions.parquet   per-observation deployment predictions
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import (FEATURES, MODEL_ORDER, RANDOM_STATE, RESULTS, TARGETS,
                    TRAIN_END_BLOCK, build, design, load_primary, metrics)

N_TREES_FIXED = 500


def model(name):
    return build(name, {"n_estimators": N_TREES_FIXED} if name == "Random forest" else None)


def main() -> None:
    df = load_primary()
    rows, preds = [], []
    t0 = time.time()
    for target in TARGETS:
        d, X, y, blk = design(df, target)
        print(f"[s02] {target}: n={len(d)}", flush=True)

        for name in MODEL_ORDER:
            # Protocol A - random 10-fold cross-validation
            kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
            for fold, (tr, te) in enumerate(kf.split(X)):
                m = model(name).fit(X[tr], y[tr])
                rows.append(dict(target=target, model=name,
                                 protocol="Random 10-fold CV", fold=fold,
                                 **metrics(y[te], m.predict(X[te]))))
            # Protocol B - forward chaining by month
            for k in range(2, 12):
                tr, te = np.where(blk <= k)[0], np.where(blk == k + 1)[0]
                if len(te) < 50:
                    continue
                m = model(name).fit(X[tr], y[tr])
                rows.append(dict(target=target, model=name,
                                 protocol="Forward-chaining CV", fold=k + 1,
                                 **metrics(y[te], m.predict(X[te]))))
            # Protocol C - deployment split
            tr = np.where(blk <= TRAIN_END_BLOCK)[0]
            te = np.where(blk > TRAIN_END_BLOCK)[0]
            m = model(name).fit(X[tr], y[tr])
            p = m.predict(X[te])
            rows.append(dict(target=target, model=name, protocol="Deployment split",
                             fold=0, **metrics(y[te], p)))
            preds.append(pd.DataFrame(dict(target=target, model=name,
                                           block=blk[te], y=y[te], pred=p)))
            print(f"       {name:24s} {time.time() - t0:6.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "model_results_raw.csv", index=False)
    pd.concat(preds).to_parquet(RESULTS / "deployment_predictions.parquet", index=False)
    print(f"[s02] wrote {len(rows)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
