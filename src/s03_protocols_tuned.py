"""Stage 3 - protocol comparison with protocol-matched nested tuning (Sect. 4.1-4.2).

The inner loop mirrors the outer protocol: a random inner split under the random
outer protocol, a blocked forward-chaining inner split under both blocked outer
protocols. This is the experiment that shows the inflation is not a tuning
artefact and that random inner validation switches regularisation off.

Runtime is roughly 30 min single-threaded.

Output
    results/model_results_tuned.csv   336 rows, with the selected grid point
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from common import (GRIDS, MODEL_ORDER, RANDOM_STATE, RESULTS, TARGETS,
                    TRAIN_END_BLOCK, build, design, load_primary, metrics)

N_INNER = 3


def inner_random(n: int, k: int = N_INNER):
    return list(KFold(k, shuffle=True, random_state=RANDOM_STATE).split(np.arange(n)))


def inner_blocked(blocks, k: int = N_INNER):
    """Forward-chaining inner split over the last *k* blocks of the training set."""
    out = []
    for b in np.unique(blocks)[-k:]:
        tr, va = np.where(blocks < b)[0], np.where(blocks == b)[0]
        if len(tr) > 100 and len(va) > 20:
            out.append((tr, va))
    return out or inner_random(len(blocks), 2)


def tune(name, X, y, blocks, blocked: bool):
    splits = inner_blocked(blocks) if blocked else inner_random(len(y))
    best = (np.inf, None)
    for params in GRIDS[name]:
        scores = []
        for tr, va in splits:
            m = build(name, params).fit(X[tr], y[tr])
            scores.append(np.sqrt(mean_squared_error(y[va], m.predict(X[va]))))
        score = float(np.mean(scores))
        if score < best[0]:
            best = (score, params)
    return build(name, best[1]).fit(X, y), best[1]


def main() -> None:
    df = load_primary()
    rows = []
    t0 = time.time()
    for target in TARGETS:
        d, X, y, blk = design(df, target)
        print(f"[s03] {target}: n={len(d)}", flush=True)
        for name in MODEL_ORDER:
            # Protocol A - random 5-fold outer (reduced from 10 for tuning cost)
            for fold, (tr, te) in enumerate(
                    KFold(5, shuffle=True, random_state=RANDOM_STATE).split(X)):
                m, params = tune(name, X[tr], y[tr], blk[tr], blocked=False)
                rows.append(dict(target=target, model=name,
                                 protocol="Random 5-fold CV (tuned)", fold=fold,
                                 params=json.dumps(params, default=str),
                                 **metrics(y[te], m.predict(X[te]))))
            # Protocol B - forward chaining
            for k in range(2, 12):
                tr, te = np.where(blk <= k)[0], np.where(blk == k + 1)[0]
                if len(te) < 50:
                    continue
                m, params = tune(name, X[tr], y[tr], blk[tr], blocked=True)
                rows.append(dict(target=target, model=name,
                                 protocol="Forward-chaining CV (tuned)", fold=k + 1,
                                 params=json.dumps(params, default=str),
                                 **metrics(y[te], m.predict(X[te]))))
            # Protocol C - deployment split
            tr = np.where(blk <= TRAIN_END_BLOCK)[0]
            te = np.where(blk > TRAIN_END_BLOCK)[0]
            m, params = tune(name, X[tr], y[tr], blk[tr], blocked=True)
            rows.append(dict(target=target, model=name,
                             protocol="Deployment split (tuned)", fold=0,
                             params=json.dumps(params, default=str),
                             **metrics(y[te], m.predict(X[te]))))
            print(f"       {name:24s} {time.time() - t0:6.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / "model_results_tuned.csv", index=False)
    print(f"[s03] wrote {len(rows)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
