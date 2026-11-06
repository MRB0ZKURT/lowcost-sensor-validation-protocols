#!/usr/bin/env python
"""Run the whole analysis pipeline in dependency order.

    python run_all.py              # every stage
    python run_all.py 4 5 6        # only the named stages
    python run_all.py --list       # show the stages and their approximate runtime

Stage 3 dominates the runtime (about 30 min single-threaded); the whole
pipeline takes roughly one hour on a laptop CPU.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

# Belt and braces: the stage scripts pin these themselves by importing
# src/common.py first, but setting them here protects against an import-order
# regression, because the limits only take effect before numpy/sklearn load.
ENV = dict(os.environ)
ENV.update({v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
ENV["JOBLIB_MULTIPROCESSING"] = "0"

STAGES = [
    (1, "s01_data.py", "download data, build analysis tables", "~1 min"),
    (2, "s02_protocols_fixed.py", "protocol comparison, fixed hyperparameters", "~8 min"),
    (3, "s03_protocols_tuned.py", "protocol comparison, nested tuning", "~31 min"),
    (4, "s04_rolling_origin.py", "rolling-origin horizon analysis", "~1 min"),
    (5, "s05_recalibration.py", "sliding-window recalibration", "~4 min"),
    (6, "s06_drift_corrections.py", "unsupervised drift corrections", "~1 min"),
    (7, "s07_importance.py", "permutation importance", "~1 min"),
    (8, "s08_uncertainty.py", "regulatory uncertainty", "~5 min"),
    (9, "s09_vergara.py", "independent-array replication", "~8 min"),
    (10, "s10_statistics.py", "statistical inference", "<1 min"),
    (11, "s11_figures.py", "figures 1-6", "<1 min"),
]


def main() -> int:
    args = sys.argv[1:]
    if "--list" in args:
        for n, f, d, t in STAGES:
            print(f"  {n:2d}  {f:28s} {d:46s} {t}")
        return 0
    wanted = {int(a) for a in args if a.isdigit()} or {n for n, *_ in STAGES}
    t0 = time.time()
    for n, fname, desc, _ in STAGES:
        if n not in wanted:
            continue
        print(f"\n=== stage {n}: {desc} ===", flush=True)
        r = subprocess.run([sys.executable, str(SRC / fname)], cwd=str(SRC), env=ENV)
        if r.returncode != 0:
            print(f"stage {n} failed with exit code {r.returncode}")
            return r.returncode
    print(f"\nall requested stages finished in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
