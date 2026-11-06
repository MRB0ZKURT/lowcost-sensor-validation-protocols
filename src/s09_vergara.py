"""Stage 9 - replication of the protocol effect on the independent array (Sect. 4.6).

Six-gas classification on the 16-sensor array, under the same three schemes
built over acquisition batches instead of calendar months.

Output
    results/vergara_protocols.csv
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from common import DATA, RANDOM_STATE, RESULTS

MODELS = ["Logistic regression", "Linear SVM", "RBF SVM", "Random forest"]


def build_clf(name: str) -> Pipeline:
    if name == "Logistic regression":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=3000))])
    if name == "Linear SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LinearSVC(C=1.0, max_iter=5000))])
    if name == "RBF SVM":
        return Pipeline([("sc", StandardScaler()), ("m", SVC(C=10.0, gamma="scale"))])
    return Pipeline([("m", RandomForestClassifier(n_estimators=300,
                                                  random_state=RANDOM_STATE, n_jobs=1))])


def score(y, p) -> dict:
    return {"acc": accuracy_score(y, p), "f1": f1_score(y, p, average="macro")}


def main() -> None:
    D = pd.read_parquet(DATA / "vergara_drift.parquet")
    feats = [c for c in D.columns if c.startswith("f")]
    X, y, bt = D[feats].to_numpy(), D.gas.to_numpy(), D.batch.to_numpy()
    print(f"[s09] {D.shape[0]} samples, batches {dict(D.groupby('batch').size())}")

    rows = []
    t0 = time.time()
    for name in MODELS:
        for fold, (tr, te) in enumerate(
                StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE).split(X, y)):
            m = build_clf(name).fit(X[tr], y[tr])
            rows.append(dict(model=name, protocol="Random 10-fold CV", fold=fold,
                             **score(y[te], m.predict(X[te]))))
        for k in range(1, 10):
            tr, te = np.where(bt <= k)[0], np.where(bt == k + 1)[0]
            m = build_clf(name).fit(X[tr], y[tr])
            rows.append(dict(model=name, protocol="Forward-chaining by batch",
                             fold=k + 1, **score(y[te], m.predict(X[te]))))
        tr = np.where(bt == 1)[0]
        m = build_clf(name).fit(X[tr], y[tr])
        for k in range(2, 11):
            te = np.where(bt == k)[0]
            rows.append(dict(model=name, protocol="Deployment (train batch 1)",
                             fold=k, **score(y[te], m.predict(X[te]))))
        print(f"       {name:20s} {time.time() - t0:6.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "vergara_protocols.csv", index=False)
    print("[s09] wrote vergara_protocols.csv")


if __name__ == "__main__":
    main()
