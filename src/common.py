"""Shared configuration, data loading and model definitions.

Importing this module must be the FIRST thing every stage script does, because it
pins the thread counts before numpy/sklearn are imported.
"""
from __future__ import annotations

import os

# --- determinism and single-threaded execution -----------------------------
# Thread limits must be set before numpy/sklearn import. Single-threaded
# execution also makes the results bitwise reproducible.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

joblib.parallel_backend("sequential").__enter__()
warnings.filterwarnings("ignore")

RANDOM_STATE = 0

# --- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # the repository root (code/)
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (DATA, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --- variables --------------------------------------------------------------
SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)",
           "PT08.S5(O3)"]
METEO = ["T", "RH", "AH"]
FEATURES = SENSORS + METEO
TARGETS = ["CO(GT)", "NOx(GT)", "NO2(GT)"]

# EU limit values used for the relative expanded uncertainty.
# NO2: 1 h limit value. CO: 8 h limit value (evaluated here on hourly data,
# therefore indicative only). NOx has no health-based limit value.
LIMIT_VALUES = {"NO2(GT)": 200.0, "CO(GT)": 10.0, "NOx(GT)": None}

# Calendar label of each monthly block (block 0 = March 2004).
BLOCK_MONTH = {0: "Mar", 1: "Apr", 2: "May", 3: "Jun", 4: "Jul", 5: "Aug",
               6: "Sep", 7: "Oct", 8: "Nov", 9: "Dec", 10: "Jan", 11: "Feb",
               12: "Mar"}

# Last block of the calibration window in the deployment protocol.
TRAIN_END_BLOCK = 5


def load_primary() -> pd.DataFrame:
    """Return the cleaned hourly co-location record with a ``block`` column."""
    df = pd.read_parquet(DATA / "airquality_clean.parquet")
    # The 87 h partial block of April 2005 is merged into the last full block.
    df["block"] = df["month_idx"].clip(upper=12)
    return df


def design(df: pd.DataFrame, target: str, extra=None):
    """Complete-case design matrix for one target.

    Returns ``(frame, X, y, blocks)``; *frame* keeps ``datetime`` so that callers
    can build time windows.
    """
    cols = FEATURES + (extra or [])
    d = df[["datetime", "block"] + cols + [target]].dropna().reset_index(drop=True)
    return d, d[cols].to_numpy(float), d[target].to_numpy(float), d["block"].to_numpy()


# --- models -----------------------------------------------------------------
MODEL_ORDER = ["Linear regression", "Quadratic ridge", "k-nearest neighbours",
               "Support vector reg.", "Random forest", "Gradient boosting",
               "Neural network (MLP)"]

# The three representative models used in the experiments that would otherwise
# be too expensive to run for all seven.
REPRESENTATIVE = ["Linear regression", "Random forest", "Neural network (MLP)"]


def build(name: str, params: dict | None = None) -> Pipeline:
    """Model factory. With ``params=None`` the fixed configuration is returned."""
    p = params or {}
    if name == "Linear regression":
        return Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    if name == "Quadratic ridge":
        # Fixed configuration (no alpha given): RidgeCV over a 13-point grid,
        # which is the conventional default for this model. When the nested
        # search supplies an alpha, a plain Ridge is fitted with that value.
        est = (Ridge(alpha=p["alpha"]) if "alpha" in p
               else RidgeCV(alphas=np.logspace(-3, 3, 13)))
        return Pipeline([("pf", PolynomialFeatures(2, include_bias=False)),
                         ("sc", StandardScaler()), ("m", est)])
    if name == "k-nearest neighbours":
        return Pipeline([("sc", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=p.get("n_neighbors", 10),
                                                   weights="distance"))])
    if name == "Support vector reg.":
        return Pipeline([("sc", StandardScaler()),
                         ("m", SVR(C=p.get("C", 10.0), gamma="scale", epsilon=0.1))])
    if name == "Random forest":
        return Pipeline([("m", RandomForestRegressor(
            n_estimators=p.get("n_estimators", 300),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            max_features=p.get("max_features", 1.0),
            random_state=RANDOM_STATE, n_jobs=1))])
    if name == "Gradient boosting":
        return Pipeline([("m", HistGradientBoostingRegressor(
            learning_rate=p.get("learning_rate", 0.1),
            max_leaf_nodes=p.get("max_leaf_nodes", 31),
            l2_regularization=p.get("l2", 0.0),
            random_state=RANDOM_STATE))])
    if name == "Neural network (MLP)":
        return Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=p.get("h", (64, 32)),
                                            alpha=p.get("alpha", 1e-4),
                                            max_iter=2000, early_stopping=True,
                                            random_state=RANDOM_STATE))])
    raise ValueError(f"unknown model: {name}")


# Hyperparameter grids for the nested search (Sect. 3.2 of the manuscript).
GRIDS = {
    "Linear regression": [{}],
    "Quadratic ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0, 1000.0)],
    "k-nearest neighbours": [{"n_neighbors": k} for k in (5, 10, 20, 50)],
    "Support vector reg.": [{"C": c} for c in (1.0, 10.0, 100.0)],
    "Random forest": [{"min_samples_leaf": leaf, "max_features": frac}
                      for leaf in (1, 5, 20) for frac in (1.0, 0.5)],
    "Gradient boosting": [{"learning_rate": lr, "max_leaf_nodes": n, "l2": l2}
                          for lr in (0.05, 0.1) for n in (15, 31) for l2 in (0.0, 1.0)],
    "Neural network (MLP)": [{"h": (32,), "alpha": 1e-3},
                             {"h": (64, 32), "alpha": 1e-3},
                             {"h": (64, 32), "alpha": 1e-2}],
}

# The least-regularised option in each grid, used in Fig. 6b.
LEAST_REGULARISED = {"Quadratic ridge": ("alpha", "0.1"),
                     "Random forest": ("min_samples_leaf", "1"),
                     "k-nearest neighbours": ("n_neighbors", "5")}


def metrics(y, pred) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
            "bias": float(np.mean(pred - y)),
            "n": int(len(y))}


def orthogonal_regression(x, y):
    """Intercept and slope of the orthogonal (total least squares) fit y = a + b x."""
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).mean()
    syy = ((y - my) ** 2).mean()
    sxy = ((x - mx) * (y - my)).mean()
    b = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    return float(my - b * mx), float(b)
