"""Stage 1 - download both public datasets and build the analysis tables.

Outputs
    data/airquality_clean.parquet   cleaned hourly co-location record
    data/vergara_drift.parquet      independent 16-sensor array dataset
    results/table_correlations.csv  Spearman correlations (Sect. 2.1)
    results/table_monthly_summary.csv  monthly means (Sect. 2.1)
"""
import common  # noqa: F401  must be first: pins thread limits before numpy/sklearn
import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import requests

from common import DATA, RESULTS, SENSORS, METEO, TARGETS

AIRQUALITY_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
VERGARA_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"


def _fetch(url: str) -> zipfile.ZipFile:
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    print(f"  {url.rsplit('/', 1)[-1]}: {len(r.content)} bytes, "
          f"sha256 {hashlib.sha256(r.content).hexdigest()[:16]}")
    return zipfile.ZipFile(io.BytesIO(r.content))


def build_airquality() -> pd.DataFrame:
    z = _fetch(AIRQUALITY_URL)
    z.extractall(DATA / "airquality_raw")
    df = pd.read_csv(DATA / "airquality_raw" / "AirQualityUCI.csv", sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    ts = pd.to_datetime(df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
                        format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).replace(-200, np.nan)
    df.insert(0, "datetime", ts)
    # NMHC(GT) is 90.2 % missing; C6H6(GT) is a near-deterministic transform of
    # PT08.S2 and is dropped to avoid a spurious target (Sect. 2.1).
    df = df.drop(columns=["NMHC(GT)"])
    df["month_idx"] = ((df.datetime.dt.year - 2004) * 12 + df.datetime.dt.month) - 3
    df["hour"] = df.datetime.dt.hour
    df["dow"] = df.datetime.dt.day_of_week
    df.to_parquet(DATA / "airquality_clean.parquet", index=False)

    span = (df.datetime.max() - df.datetime.min()).total_seconds() / 3600 + 1
    print(f"  rows={len(df)}  span={span:.0f} h  duplicate timestamps="
          f"{int(df.datetime.duplicated().sum())}")
    return df


def build_vergara() -> pd.DataFrame:
    z = _fetch(VERGARA_URL)
    z.extractall(DATA / "vergara_raw")
    rows = []
    for b in range(1, 11):
        path = DATA / "vergara_raw" / "Dataset" / f"batch{b}.dat"
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0].split(";")[0])
                v = np.zeros(128)
                for f in parts[1:]:
                    k, val = f.split(":")
                    v[int(k) - 1] = float(val)
                rows.append((b, cls, *v))
    d = pd.DataFrame(rows, columns=["batch", "gas"] + [f"f{i+1}" for i in range(128)])
    d.to_parquet(DATA / "vergara_drift.parquet", index=False)
    print(f"  rows={len(d)}  batches={dict(d.groupby('batch').size())}")
    return d


def descriptive_tables(df: pd.DataFrame) -> None:
    rows = []
    for t in TARGETS:
        for s in SENSORS + METEO:
            sub = df[[t, s]].dropna()
            rows.append({"target": t, "feature": s,
                         "spearman": sub[t].corr(sub[s], method="spearman")})
    corr = pd.DataFrame(rows).pivot(index="feature", columns="target", values="spearman")
    corr.round(3).to_csv(RESULTS / "table_correlations.csv")

    mon = df.groupby("block" if "block" in df else "month_idx").agg(
        S4=("PT08.S4(NO2)", "mean"), NO2=("NO2(GT)", "mean"),
        S3=("PT08.S3(NOx)", "mean"), NOx=("NOx(GT)", "mean"),
        S1=("PT08.S1(CO)", "mean"), CO=("CO(GT)", "mean"),
        T=("T", "mean"), RH=("RH", "mean"), n=("datetime", "size")).round(1)
    mon["S4_per_NO2"] = (mon.S4 / mon.NO2).round(2)
    mon.to_csv(RESULTS / "table_monthly_summary.csv")
    print("  wrote table_correlations.csv, table_monthly_summary.csv")


if __name__ == "__main__":
    print("[s01] air quality")
    aq = build_airquality()
    print("[s01] vergara")
    build_vergara()
    print("[s01] descriptive tables")
    descriptive_tables(aq)
