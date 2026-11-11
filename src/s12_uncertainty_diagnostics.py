"""Stage 12 - diagnostics for the regulatory uncertainty calculation (Sect. 3.5).

Two checks that the reported relative expanded uncertainties do not depend on
choices made silently in stage 8:

1. Sensitivity to the reference uncertainty. Stage 8 sets u(x) = 0 in
   Eq. (8.37) of the equivalence guide, which makes the reported REU an upper
   bound. Here the REU is recomputed over a grid of u(x) running from zero to
   the largest value the reference method may have while still meeting the
   data-quality objective for FIXED measurements of Directive (EU) 2024/2881
   (Annex V, Table 2): 15 % of the limit value expanded, i.e. u(x) = 15 ug/m3
   for hourly NO2 and 0.5 mg/m3 for the eight-hour CO mean (k = 2). The value
   of u(x) at which the REU would cross the indicative objective is reported.

2. Choice between Eqs. (8.38a) and (8.38b). The guide offers an absolute and a
   relative residual sum of squares, to be used according to whether the
   residuals or the relative residuals are constant with concentration. The
   residuals are binned by reference concentration and the dispersion of both
   quantities across bins is compared; the REU is then recomputed with
   Eq. (8.38b) so that the consequence of the choice is visible.

Outputs
    results/table_reu_sensitivity.csv
    results/table_residual_scaling.csv
    figures/figA1_residual_diagnostics.png
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, LIMIT_VALUES, RESULTS, orthogonal_regression

# standard uncertainty a compliant reference analyser may have at the limit
# value: half the expanded data-quality objective for fixed measurements
U_REF_MAX = {"NO2(GT)": 15.0, "CO(GT)": 0.5}
DQO_INDICATIVE = {"NO2(GT)": 25.0, "CO(GT)": 20.0}
N_BINS = 10
PLOT = [("NO2(GT)", "Nitrogen dioxide", "reference NO$_2$ (\u03bcg m$^{-3}$)"),
        ("CO(GT)", "Carbon monoxide", "reference CO (mg m$^{-3}$)")]
PROTOCOLS = ["Random 10-fold CV", "Forward-chaining CV", "Deployment split"]


def reu(rss, n, a, b, lv, u_ref=0.0):
    """Eq. (8.37) evaluated at the limit value, expanded with k = 2."""
    var = rss / (n - 2) - u_ref ** 2 + (a + (b - 1) * lv) ** 2
    return float("nan") if var <= 0 else 200 * np.sqrt(var) / lv


def crossing(rss, n, a, b, lv, target):
    """u(x) at which the REU falls to the indicative objective, if reachable."""
    need = (DQO_INDICATIVE[target] * lv / 200) ** 2
    u2 = rss / (n - 2) + (a + (b - 1) * lv) ** 2 - need
    return float(np.sqrt(u2)) if u2 > 0 else 0.0


def main() -> None:
    P = pd.read_parquet(RESULTS / "predictions_all_protocols.parquet")
    sens, scal, resid = [], [], {}
    for (target, model, protocol), g in P.groupby(["target", "model", "protocol"]):
        lv = LIMIT_VALUES[target]
        if lv is None:
            continue
        x, y = g.y.to_numpy(), g.pred.to_numpy()
        n = len(x)
        a, b = orthogonal_regression(x, y)
        r = y - a - b * x
        rss_abs = float((r ** 2).sum())
        fit = a + b * x
        ok = np.abs(fit) > 1e-9
        # Eq. (8.38b) is a relative sum of squares; evaluated at the limit value
        # the corresponding variance term is (RSS_rel/(n-2)) * LV^2
        rss_rel = float(((r[ok] / fit[ok]) ** 2).sum()) * lv ** 2

        umax = U_REF_MAX[target]
        row = dict(target=target, model=model, protocol=protocol, n=n, slope=b, intercept=a)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            row[f"REU_u{int(frac*100):03d}"] = reu(rss_abs, n, a, b, lv, frac * umax)
        row["u_ref_max"] = umax
        row["u_ref_crossing"] = crossing(rss_abs, n, a, b, lv, target)
        row["crossing_frac_of_max"] = row["u_ref_crossing"] / umax
        row["REU_eq838b"] = reu(rss_rel, n, a, b, lv)
        sens.append(row)

        # dispersion of residuals and relative residuals across concentration bins
        q = pd.qcut(x, N_BINS, labels=False, duplicates="drop")
        d = pd.DataFrame(dict(bin=q, r=r, rel=np.where(ok, r / np.where(ok, fit, 1), np.nan)))
        s_abs = d.groupby("bin").r.std()
        s_rel = d.groupby("bin").rel.std()
        scal.append(dict(target=target, model=model, protocol=protocol,
                         cv_abs=float(s_abs.std() / s_abs.mean()),
                         cv_rel=float(s_rel.std() / s_rel.mean()),
                         favours="8.38a" if s_abs.std() / s_abs.mean()
                                 < s_rel.std() / s_rel.mean() else "8.38b"))
        resid[(target, model, protocol)] = (x, r)

    pd.DataFrame(sens).to_csv(RESULTS / "table_reu_sensitivity.csv", index=False)
    pd.DataFrame(scal).to_csv(RESULTS / "table_residual_scaling.csv", index=False)

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "figure.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans"})
    blue, orange, dark = "#4878CF", "#EE854A", "#333333"
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    for col, (target, title, xlab) in enumerate(PLOT):
        for row, proto in enumerate(["Random 10-fold CV", "Deployment split"]):
            ax = axes[row, col]
            x, r = resid[(target, "Random forest", proto)]
            ax.scatter(x, r, s=2, alpha=0.25, color=blue if row == 0 else dark,
                       edgecolors="none", rasterized=True)
            q = pd.qcut(x, N_BINS, labels=False, duplicates="drop")
            d = pd.DataFrame(dict(bin=q, x=x, r=r))
            m = d.groupby("bin").agg(x=("x", "mean"), s=("r", "std"))
            ax.plot(m.x, m.s, color=orange, lw=1.4, marker="o", ms=3,
                    label="standard deviation of residuals in bin")
            ax.plot(m.x, -m.s, color=orange, lw=1.4, marker="o", ms=3)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(f"{title} - {proto.lower()}", fontsize=7.5)
            ax.set_ylabel("residual" if col == 0 else "")
            if row == 1:
                ax.set_xlabel(xlab)
            ax.grid(axis="y", color="0.92", lw=0.5)
    axes[0, 0].legend(frameon=False, loc="upper left", handletextpad=0.4)
    fig.tight_layout(w_pad=0.9, h_pad=0.8)
    fig.savefig(FIGURES / "figA1_residual_diagnostics.png")
    plt.close(fig)
    print("[s12] wrote table_reu_sensitivity.csv, table_residual_scaling.csv, figA1")


if __name__ == "__main__":
    main()
