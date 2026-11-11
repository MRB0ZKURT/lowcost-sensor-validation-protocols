"""Stage 13 - inner validation scheme under a common outer test (Sect. 4.2).

In stage 3 the inner and the outer split change together, so the inner scheme
cannot be separated from the test set it is scored on. Here the outer test is
held fixed at the deployment split: both inner schemes select hyperparameters
on exactly the same training half, the selected configuration is refitted on
that whole training half, and both are scored on exactly the same deployment
test half. Any difference is attributable to the inner scheme alone.

Only the three models whose grid contains an explicit regularisation-like knob
are run, which are the three shown in Fig. 2b.

Output
    results/table_inner_validation.csv
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import json

import numpy as np
import pandas as pd

from common import (RESULTS, TARGETS, TRAIN_END_BLOCK, build, design,
                    load_primary, metrics)
from s03_protocols_tuned import tune

MODELS = ["Quadratic ridge", "Random forest", "k-nearest neighbours"]


def main() -> None:
    df = load_primary()
    rows = []
    for target in TARGETS:
        d, X, y, blk = design(df, target)
        tr = np.where(blk <= TRAIN_END_BLOCK)[0]
        te = np.where(blk > TRAIN_END_BLOCK)[0]
        for name in MODELS:
            for label, blocked in (("random inner CV", False), ("blocked inner CV", True)):
                m, params = tune(name, X[tr], y[tr], blk[tr], blocked=blocked)
                rows.append(dict(target=target, model=name, inner=label,
                                 params=json.dumps(params, default=str),
                                 n_train=len(tr), n_test=len(te),
                                 **metrics(y[te], m.predict(X[te]))))
                print(f"[s13] {target} {name} {label}: {params} "
                      f"rmse={rows[-1]['rmse']:.4g}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "table_inner_validation.csv", index=False)
    w = out.pivot_table(index=["target", "model"], columns="inner", values="rmse")
    w["ratio_random_over_blocked"] = w["random inner CV"] / w["blocked inner CV"]
    print(w.round(4).to_string())


if __name__ == "__main__":
    main()
