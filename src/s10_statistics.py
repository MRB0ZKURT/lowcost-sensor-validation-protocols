"""Stage 10 - statistical inference (Sect. 3.6) and the numbers quoted in the text.

Outputs
    results/table_horizon_effect.csv
    results/table_protocol_inflation.csv
    results/table_bootstrap_ci.csv
    results/table_tuned_summary.csv
    results/statistics_report.txt
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import io

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import mean_squared_error

from common import LEAST_REGULARISED, MODEL_ORDER, RESULTS

OUT = io.StringIO()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    OUT.write(line + "\n")


def cluster_quantile(n_clusters: int) -> float:
    """Two-sided 97.5 % quantile of t with G-1 degrees of freedom.

    Cluster-robust standard errors are asymptotic in the number of clusters.
    With G as small as the seven training origins of the rolling-origin design,
    the normal quantile is anti-conservative, so every interval and p value
    computed from a clustered covariance below uses t(G-1) instead.
    """
    return float(stats.t.ppf(0.975, n_clusters - 1))


def cluster_pvalue(b: float, se: float, n_clusters: int) -> float:
    """Two-sided p value of b/se against t(G-1)."""
    return float(2 * stats.t.sf(abs(b / se), n_clusters - 1))


def horizon_effect() -> None:
    ro = pd.read_csv(RESULTS / "rolling_origin.csv")
    ro["lr"] = np.log(ro.rmse)
    say("\n[horizon effect: log(RMSE) ~ horizon + C(test_month) + C(model),"
        " SE clustered by training origin]")
    rows = []
    for target, g in ro.groupby("target"):
        m = smf.ols("lr ~ horizon + C(test_month) + C(model)", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g["origin"]})
        b, se = m.params["horizon"], m.bse["horizon"]
        G = int(g["origin"].nunique())
        tq = cluster_quantile(G)
        p = cluster_pvalue(b, se, G)
        lo, hi = b - tq * se, b + tq * se
        mc = {k: v for k, v in m.params.items() if k.startswith("C(test_month)")}
        season = 100 * (np.exp(max(mc.values()) - min(mc.values())) - 1)
        rows.append(dict(target=target, pct_per_month=100 * (np.exp(b) - 1),
                         lo=100 * (np.exp(lo) - 1), hi=100 * (np.exp(hi) - 1),
                         p=p, season_spread_pct=season, n=len(g), n_clusters=G))
        say(f"  {target:9s} +{100*(np.exp(b)-1):5.1f} %/month "
            f"(95 % CI {100*(np.exp(lo)-1):5.1f} to {100*(np.exp(hi)-1):5.1f}), "
            f"p={p:.3g} | season spread {season:5.0f} %")
    pd.DataFrame(rows).to_csv(RESULTS / "table_horizon_effect.csv", index=False)

    say("\n[horizon slope, leaving out one training origin at a time]")
    for target, g in ro.groupby("target"):
        slopes = []
        for o in sorted(g["origin"].unique()):
            gg = g[g["origin"] != o]
            mm = smf.ols("lr ~ horizon + C(test_month) + C(model)", data=gg).fit(
                cov_type="cluster", cov_kwds={"groups": gg["origin"]})
            slopes.append(100 * (np.exp(mm.params["horizon"]) - 1))
        say(f"  {target:9s} {min(slopes):5.1f} to {max(slopes):5.1f} %/month "
            f"over the {len(slopes)} leave-one-origin-out fits")

    say("\n[per-model horizon slope]")
    for target in ["NO2(GT)", "NOx(GT)"]:
        for mod, g in ro[ro.target == target].groupby("model"):
            m = smf.ols("lr ~ horizon + C(test_month)", data=g).fit(
                cov_type="cluster", cov_kwds={"groups": g["origin"]})
            p = cluster_pvalue(m.params["horizon"], m.bse["horizon"],
                               int(g["origin"].nunique()))
            say(f"  {target:9s} {mod:22s} +{100*(np.exp(m.params['horizon'])-1):5.1f} "
                f"%/month (p={p:.3g})")


def protocol_inflation(path: str, reference: str, label: str) -> None:
    r = pd.read_csv(RESULTS / path)
    r["lr"] = np.log(r.rmse)
    r["cl"] = r.target + "|" + r.model
    m = smf.ols(f"lr ~ C(protocol, Treatment('{reference}')) + C(target) + C(model)",
                data=r).fit(cov_type="cluster", cov_kwds={"groups": r["cl"]})
    say(f"\n[{label}: OLS on log(RMSE), SE clustered by target-model cell]")
    for k in m.params.index:
        if "protocol" in k:
            b, se = m.params[k], m.bse[k]
            G = int(r["cl"].nunique())
            tq = cluster_quantile(G)
            say(f"  {k.split('T.')[1].rstrip(']'):32s} x{np.exp(b):.2f} "
                f"(95 % CI {np.exp(b-tq*se):.2f}-{np.exp(b+tq*se):.2f}), "
                f"p={cluster_pvalue(b, se, G):.3g}")
    g = r.pivot_table(index=["target", "model"], columns="protocol",
                      values="rmse", aggfunc="mean")
    for alt in [c for c in g.columns if c != reference]:
        g[f"ratio: {alt}"] = g[alt] / g[reference]
    g.round(3).to_csv(RESULTS / ("table_protocol_inflation.csv" if "raw" in path
                                 else "table_tuned_summary.csv"))
    return g, reference


def ranking_and_tests(g, reference) -> None:
    say("\n[rank agreement between protocols, Spearman rho over the 7 models]")
    for target in g.index.get_level_values(0).unique():
        s = g.loc[target]
        for alt in [c for c in s.columns if c != reference and not c.startswith("ratio")]:
            rho = stats.spearmanr(s[reference], s[alt]).statistic
            say(f"  {target:9s} {reference[:18]:18s} vs {alt:28s} rho={rho:+.2f}")
    say("\n[per-target Wilcoxon over the 7 model means]")
    for target in g.index.get_level_values(0).unique():
        s = g.loc[target]
        for alt in [c for c in s.columns if c != reference and not c.startswith("ratio")]:
            w = stats.wilcoxon(s[alt], s[reference], alternative="greater")
            say(f"  {target:9s} {alt:28s} W={w.statistic:.0f} p={w.pvalue:.4f} "
                f"ratio median={np.median(s[alt]/s[reference]):.2f}")


def friedman() -> None:
    r = pd.read_csv(RESULTS / "model_results_raw.csv")
    say("\n[Friedman across models within protocol]")
    for target in r.target.unique():
        for proto in ["Random 10-fold CV", "Forward-chaining CV"]:
            sub = r[(r.target == target) & (r.protocol == proto)].pivot_table(
                index="fold", columns="model", values="rmse")
            st, p = stats.friedmanchisquare(*[sub[c].to_numpy() for c in sub.columns])
            say(f"  {target:9s} {proto:22s} chi2={st:.1f} p={p:.3g}")


def bootstrap_ci(B: int = 1000, L: int = 24) -> None:
    pred = pd.read_parquet(RESULTS / "deployment_predictions.parquet")
    rng = np.random.default_rng(0)
    rows = []
    for (target, mod), g in pred.groupby(["target", "model"]):
        e = (g.pred.to_numpy() - g.y.to_numpy()) ** 2
        n = len(e)
        nb = int(np.ceil(n / L))
        starts = rng.integers(0, n - L, size=(B, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
        rmses = np.sqrt(e[idx].mean(axis=1))
        lo, hi = np.percentile(rmses, [2.5, 97.5])
        rows.append(dict(target=target, model=mod, rmse=float(np.sqrt(e.mean())),
                         lo=float(lo), hi=float(hi)))
    bt = pd.DataFrame(rows)
    bt.to_csv(RESULTS / "table_bootstrap_ci.csv", index=False)
    say(f"\n[deployment RMSE, 95 % moving-block bootstrap CI (L={L} h, B={B})]")
    say(bt.round(2).to_string(index=False))


def hyperparameter_selection() -> None:
    import json
    t = pd.read_csv(RESULTS / "model_results_tuned.csv")
    say("\n[frequency of the least-regularised grid point by inner protocol]")
    for mod, (key, least) in LEAST_REGULARISED.items():
        for proto in ["Random 5-fold CV (tuned)", "Forward-chaining CV (tuned)"]:
            s = t[(t.model == mod) & (t.protocol == proto)].params.apply(
                lambda j: str(json.loads(j)[key]))
            say(f"  {mod:22s} {proto:30s} {key}={least}: {int((s==least).sum())}/{len(s)}")


def main() -> None:
    horizon_effect()
    g, ref = protocol_inflation("model_results_raw.csv", "Random 10-fold CV",
                                "protocol inflation, fixed hyperparameters")
    ranking_and_tests(g, ref)
    gt, reft = protocol_inflation("model_results_tuned.csv", "Random 5-fold CV (tuned)",
                                  "protocol inflation, nested tuning")
    ranking_and_tests(gt, reft)
    friedman()
    bootstrap_ci()
    hyperparameter_selection()
    (RESULTS / "statistics_report.txt").write_text(OUT.getvalue(), encoding="utf-8")
    print("\n[s10] wrote statistics_report.txt")


if __name__ == "__main__":
    main()
