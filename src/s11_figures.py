"""Stage 11 - all six manuscript figures.

Outputs figures/fig1 ... figures/fig6 at 300 dpi.
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, LEAST_REGULARISED, MODEL_ORDER, RESULTS, REPRESENTATIVE

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "figure.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans"})

BLUE, ORANGE, DARK, GREEN, RED = "#4878CF", "#EE854A", "#333333", "#6ACC64", "#B5443A"
PANELS = [("NO2(GT)", "Nitrogen dioxide", "RMSE (\u03bcg m$^{-3}$)"),
          ("NOx(GT)", "Nitrogen oxides", "RMSE (ppb)"),
          ("CO(GT)", "Carbon monoxide", "RMSE (mg m$^{-3}$)")]
SHORT = {m: m.replace("Support vector reg.", "Support vector machine")
          .replace("Neural network (MLP)", "Neural network") for m in MODEL_ORDER}
MONTHS = {6: "Sep", 7: "Oct", 8: "Nov", 9: "Dec", 10: "Jan", 11: "Feb", 12: "Mar"}


def fig1() -> None:
    r = pd.read_csv(RESULTS / "model_results_raw.csv")
    g = r.pivot_table(index=["target", "model"], columns="protocol",
                      values="rmse", aggfunc="mean")
    proto = [("Random 10-fold CV", "random 10-fold CV", BLUE, "o"),
             ("Forward-chaining CV", "forward-chaining CV", ORANGE, "s"),
             ("Deployment split", "deployment split (6 mo train)", DARK, "D")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    y = np.arange(len(MODEL_ORDER))
    for ax, (tgt, title, xlab) in zip(axes, PANELS):
        sub = g.loc[tgt].reindex(MODEL_ORDER)
        for i in y:
            ax.plot([sub.iloc[i]["Random 10-fold CV"], sub.iloc[i]["Deployment split"]],
                    [i, i], color="0.85", lw=1.2, zorder=1)
        for col, lab, c, mk in proto:
            ax.scatter(sub[col], y, color=c, marker=mk, s=22, zorder=3, label=lab,
                       edgecolor="white", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_ylim(-0.6, len(MODEL_ORDER) - 0.4)
        ax.invert_yaxis()
        ax.set_yticklabels([SHORT[m] for m in MODEL_ORDER] if ax is axes[0] else [])
        ax.set_xlabel(xlab)
        ax.set_title(title)
        ax.set_xlim(left=0)
        ax.grid(axis="x", color="0.92", lw=0.5, zorder=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), handletextpad=0.3)
    axes[2].annotate("lower = better", xy=(0.97, 1.14), xycoords="axes fraction",
                     ha="right", fontsize=7, color=DARK)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGURES / "fig1_protocol_optimism.png")
    plt.close(fig)


def fig2() -> None:
    base = pd.read_csv(RESULTS / "table_drift_by_month.csv")
    rc = pd.read_csv(RESULTS / "recalibration_results.csv")
    piv = pd.read_csv(RESULTS / "table_recalibration.csv", index_col=[0, 1])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for j, (tgt, title, ylab) in enumerate(PANELS):
        ax = axes[0, j]
        b = base[(base.target == tgt) & (base.model == "Random forest")].sort_values("block")
        a = rc[(rc.target == tgt) & (rc.model == "Random forest") &
               (rc.window_h == -1)].sort_values("test_block")
        ax.plot(b.block, b.rmse, color=DARK, marker="o", ms=3.5, lw=1.3,
                label="fixed calibration, never updated")
        ax.plot(a.test_block, a.rmse, color=BLUE, marker="s", ms=3.5, lw=1.3,
                label="retrained every month")
        ax.set_xticks(list(MONTHS))
        ax.set_xticklabels([MONTHS[k] for k in MONTHS])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylab if j == 0 else "")
        ax.grid(axis="y", color="0.92", lw=0.5)
        if j == 0:
            ax.legend(frameon=False, loc="upper left", handletextpad=0.4)
    axes[0, 1].set_xlabel("deployment month (Sep 2004 - Mar 2005)")

    windows = [168, 336, 720, 2160, 4320, -1]
    labels = ["1 wk", "2 wk", "1 mo", "3 mo", "6 mo", "all\nhistory"]
    styles = [(REPRESENTATIVE[0], BLUE, "o"), (REPRESENTATIVE[1], ORANGE, "s"),
              (REPRESENTATIVE[2], DARK, "^")]
    for j, (tgt, _, _) in enumerate(PANELS):
        ax = axes[1, j]
        ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
        for mname, c, mk in styles:
            row = piv.loc[(tgt, mname)]
            vals = [row["all past" if w == -1 else f"{w} h"] / row["never updated"]
                    for w in windows]
            ax.plot(range(len(windows)), vals, color=c, marker=mk, ms=3.5, lw=1.3,
                    label=SHORT[mname] if j == 0 else None)
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("RMSE relative to\nno retraining" if j == 0 else "")
        ax.grid(axis="y", color="0.92", lw=0.5)
        if j == 0:
            ax.legend(frameon=False, loc="upper right", handletextpad=0.4)
    axes[1, 0].annotate("no retraining", xy=(0.55, 1.0),
                        xycoords=("axes fraction", "data"), fontsize=6,
                        color="0.45", va="bottom")
    axes[1, 1].set_xlabel("length of recent reference data used for retraining")
    fig.tight_layout(h_pad=1.4, w_pad=0.8)
    fig.savefig(FIGURES / "fig2_drift_and_recalibration.png")
    plt.close(fig)


def fig3() -> None:
    imp = pd.read_csv(RESULTS / "permutation_importance.csv")
    order = ["PT08.S2(NMHC)", "PT08.S5(O3)", "PT08.S1(CO)", "PT08.S3(NOx)",
             "PT08.S4(NO2)", "T", "RH", "AH"]
    nice = {"PT08.S1(CO)": "S1 (CO sensor)", "PT08.S2(NMHC)": "S2 (hydrocarbon sensor)",
            "PT08.S3(NOx)": "S3 (NOx sensor)", "PT08.S4(NO2)": "S4 (NO2 sensor)",
            "PT08.S5(O3)": "S5 (ozone sensor)", "T": "temperature",
            "RH": "relative humidity", "AH": "absolute humidity"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    h = 0.38
    yy = np.arange(len(order))
    for ax, (tgt, title, xlab) in zip(axes, PANELS):
        d = imp[imp.target == tgt].set_index(["split", "feature"])
        a = [d.loc[("train period (in-regime)", f), "importance"] for f in order]
        b = [d.loc[("deployment period (out-of-regime)", f), "importance"] for f in order]
        ax.barh(yy - h / 2, a, height=h, color=BLUE, label="training period")
        ax.barh(yy + h / 2, b, height=h, color=ORANGE, label="deployment period")
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_yticks(yy)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticklabels([nice[f] for f in order] if ax is axes[0] else [])
        ax.set_title(title)
        ax.set_xlabel("increase in " + xlab.split(" (")[0] + " when shuffled")
        ax.grid(axis="x", color="0.92", lw=0.5)
    axes[0].legend(frameon=False, loc="lower right", handletextpad=0.4)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGURES / "fig3_permutation_importance.png")
    plt.close(fig)


def fig4() -> None:
    ro = pd.read_csv(RESULTS / "rolling_origin.csv")
    he = pd.read_csv(RESULTS / "table_horizon_effect.csv").set_index("target")
    styles = [(REPRESENTATIVE[0], BLUE, "o"), (REPRESENTATIVE[1], ORANGE, "s"),
              (REPRESENTATIVE[2], DARK, "^")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (tgt, title, ylab) in zip(axes, PANELS):
        for name, c, mk in styles:
            g = ro[(ro.target == tgt) & (ro.model == name)].groupby("horizon").rmse
            mu, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.errorbar(mu.index, mu.to_numpy(), yerr=se.to_numpy(), color=c,
                        marker=mk, ms=4, lw=1.3, capsize=2, label=SHORT[name])
        r = he.loc[tgt]
        txt = f"+{r.pct_per_month:.1f}%/month" + ("" if r.p < 0.05 else " (n.s.)")
        ax.annotate(txt, xy=(0.04, 0.93), xycoords="axes fraction", fontsize=7,
                    color=DARK, va="top")
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("months since calibration")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="0.92", lw=0.5)
    axes[0].legend(frameon=False, loc="lower right", handletextpad=0.4)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(FIGURES / "fig4_horizon_effect.png")
    plt.close(fig)


def fig5() -> None:
    reu = pd.read_csv(RESULTS / "table_reu.csv")
    proto = [("Random 10-fold CV", BLUE), ("Forward-chaining CV", ORANGE),
             ("Deployment split", DARK)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    spec = [("NO2(GT)", "Nitrogen dioxide (limit value 200 \u03bcg m$^{-3}$)"),
            ("CO(GT)", "Carbon monoxide (limit value 10 mg m$^{-3}$)")]
    for ax, (tgt, title) in zip(axes, spec):
        x = np.arange(len(REPRESENTATIVE))
        w = 0.26
        for i, (pr, c) in enumerate(proto):
            v = [reu[(reu.target == tgt) & (reu.model == m) &
                     (reu.protocol == pr)].REU_pct_at_LV.iloc[0] for m in REPRESENTATIVE]
            ax.bar(x + (i - 1) * w, v, width=w, color=c,
                   label=pr if ax is axes[0] else None)
        ax.axhline(25, color=RED, ls="--", lw=1.1,
                   label="data-quality objective,\nindicative measurements (25%)"
                   if ax is axes[0] else None)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m].replace(" ", "\n") for m in REPRESENTATIVE])
        ax.set_title(title, fontsize=7.5)
        ax.set_ylabel("relative expanded uncertainty (%)" if ax is axes[0] else "")
        ax.grid(axis="y", color="0.92", lw=0.5)
    axes[0].legend(frameon=False, loc="upper left", handletextpad=0.4)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(FIGURES / "fig5_regulatory_uncertainty.png")
    plt.close(fig)


def fig6() -> None:
    vg = pd.read_csv(RESULTS / "vergara_protocols.csv")
    tu = pd.read_csv(RESULTS / "model_results_tuned.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    protos = ["Random 10-fold CV", "Forward-chaining by batch", "Deployment (train batch 1)"]
    labels = ["random\n10-fold CV", "forward-chaining\nby batch", "deployment\n(train batch 1)"]
    cols = {"Logistic regression": BLUE, "Linear SVM": ORANGE, "RBF SVM": DARK,
            "Random forest": GREEN}
    for mod, c in cols.items():
        v = [vg[(vg.model == mod) & (vg.protocol == p)].acc.mean() for p in protos]
        ax.plot(range(3), v, color=c, marker="o", ms=4, lw=1.3, label=mod)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("classification accuracy")
    ax.set_title("Independent array: 16 sensors, 36 months", fontsize=7.5)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.4)

    ax = axes[1]
    inner = [("Random 5-fold CV (tuned)", "random inner CV", BLUE),
             ("Forward-chaining CV (tuned)", "blocked inner CV", ORANGE)]
    x = np.arange(len(LEAST_REGULARISED))
    w = 0.34
    for i, (proto, lab, c) in enumerate(inner):
        frac = []
        for mod, (key, least) in LEAST_REGULARISED.items():
            s = tu[(tu.model == mod) & (tu.protocol == proto)].params.apply(
                lambda j: str(json.loads(j)[key]))
            frac.append((s == least).mean())
        ax.bar(x + (i - 0.5) * w, frac, width=w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["quadratic ridge\n(\u03b1 = 0.1)", "random forest\n(leaf = 1)",
                        "k-NN\n(k = 5)"])
    ax.set_ylabel("folds selecting the\nleast-regularised setting")
    ax.set_ylim(0, 1.08)
    ax.set_title("Which hyperparameters the inner loop picks", fontsize=7.5)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGURES / "fig6_replication_and_tuning.png")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        fn()
        print(f"[s11] {fn.__name__} done")
