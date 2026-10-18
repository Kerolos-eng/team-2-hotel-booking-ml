"""Step 2 (EDA): summary table + figures. Figures go to outputs/eda/, tables to outputs/tables/.

EDA runs on the RAW file (to document what the cleaning rules fix) and on the cleaned data
(for distributions / relationships)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import path_of
from src.data.cleaning import clean
from src.data.loader import load_raw, strip_text
from src.features.feature_sets import leakage_table

BLUE, ORANGE, RED, GREEN = "#2b5d9b", "#e07b39", "#c0392b", "#2e8b57"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def _save(fig, cfg, name):
    fig.tight_layout()
    fig.savefig(path_of(cfg, "eda_dir") / name, dpi=150)
    plt.close(fig)


def summary(raw, cfg):
    guests = raw["Adults"] + raw["Children"].fillna(0) + raw["Babies"]
    nights = raw["StaysInWeekendNights"] + raw["StaysInWeekNights"]
    items = {
        "rows": len(raw), "columns": raw.shape[1],
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "missing_Children (NaN)": int(raw["Children"].isna().sum()),
        "missing_Country (NaN)": int(raw["Country"].isna().sum()),
        "Agent == 'NULL' (no agent)": int((raw["Agent"] == "NULL").sum()),
        "Company == 'NULL' (no company)": int((raw["Company"] == "NULL").sum()),
        "ADR == 0": int((raw["ADR"] == 0).sum()),
        "ADR < 0": int((raw["ADR"] < 0).sum()),
        "ADR >= 1000 (outlier)": int((raw["ADR"] >= cfg["cleaning"]["adr_max"]).sum()),
        "ADR max": float(raw["ADR"].max()),
        "zero guests": int((guests == 0).sum()),
        "zero nights": int((nights == 0).sum()),
        "cancel rate (raw)": round(float(raw["IsCanceled"].mean()), 4),
        "Agent distinct IDs": int(raw["Agent"].nunique()),
        "Company distinct IDs": int(raw["Company"].nunique()),
    }
    return pd.DataFrame({"check": list(items), "value": list(items.values())})


def column_profile(raw):
    """Profile every source variable for type, missingness, cardinality and value range."""
    rows = []
    for col in raw.columns:
        series = raw[col]
        row = {
            "column": col,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "distinct_count": int(series.nunique(dropna=True)),
            "null_token_count": int(series.eq("NULL").sum()) if pd.api.types.is_string_dtype(series.dtype) else 0,
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update({
                "min": series.min(), "median": series.median(), "mean": series.mean(),
                "max": series.max(), "top_value": None, "top_frequency": None,
            })
        else:
            counts = series.value_counts(dropna=True)
            row.update({
                "min": None, "median": None, "mean": None, "max": None,
                "top_value": counts.index[0] if not counts.empty else None,
                "top_frequency": int(counts.iloc[0]) if not counts.empty else 0,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg):
    raw = strip_text(load_raw(cfg))
    cdf, report = clean(raw, cfg["cleaning"]["adr_max"])

    s = summary(raw, cfg)
    s.to_csv(path_of(cfg, "table_dir") / "eda_summary.csv", index=False)
    column_profile(raw).to_csv(path_of(cfg, "table_dir") / "column_profile.csv", index=False)
    pd.Series(report).to_csv(path_of(cfg, "table_dir") / "cleaning_report.csv", header=["value"],
                             index_label="step")
    leakage_table().to_csv(path_of(cfg, "table_dir") / "leakage_map.csv", index=False)
    print(s.to_string(index=False))
    print("\nCleaning report")
    print(pd.Series(report).to_string())

    # 1) target balance before / after cleaning
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    vals = [report["raw_cancel_rate"], report["clean_cancel_rate"]]
    bars = ax.bar(["raw (79,330)", f"clean ({report['clean_rows']:,})"], vals, color=[BLUE, ORANGE])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center")
    ax.set(ylim=(0, 0.6), ylabel="Cancel rate", title="IsCanceled balance")
    _save(fig, cfg, "01_target_balance.png")

    # 2) missing values (NaN) and 'NULL'-coded values
    miss = pd.Series({
        "Children (NaN)": raw["Children"].isna().sum(), "Country (NaN)": raw["Country"].isna().sum(),
        "Agent ('NULL')": (raw["Agent"] == "NULL").sum(), "Company ('NULL')": (raw["Company"] == "NULL").sum()})
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(miss.index[::-1], miss.values[::-1], color=[BLUE, BLUE, ORANGE, ORANGE])
    for i, v in enumerate(miss.values[::-1]):
        ax.text(v, i, f" {v:,}", va="center")
    ax.set(xlabel="Rows", title="Missing / 'NULL' values (raw)")
    ax.set_xlim(0, miss.max() * 1.18)
    _save(fig, cfg, "02_missing_values.png")

    # 3) ADR distribution (raw, log y -> shows the ~5400 outlier) and cleaned boxplot
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(raw["ADR"], bins=100, color=BLUE)
    ax[0].set_yscale("log")
    ax[0].set(xlabel="ADR", ylabel="Bookings (log scale)", title="ADR, raw (outlier at 5400)")
    ax[1].boxplot(cdf["ADR"], vert=False, flierprops={"markersize": 2, "alpha": 0.3})
    ax[1].set(xlabel="ADR", yticks=[], title="ADR, cleaned")
    _save(fig, cfg, "03_adr_distribution.png")

    # 4) ADR by arrival month
    fig, ax = plt.subplots(figsize=(9, 3.8))
    data = [cdf.loc[cdf["ArrivalDateMonth"] == m, "ADR"] for m in MONTHS]
    ax.boxplot(data, tick_labels=[m[:3] for m in MONTHS], showfliers=False)
    ax.set(ylabel="ADR", title="ADR by arrival month (cleaned)")
    _save(fig, cfg, "04_adr_by_month.png")

    # 5) lead time: histogram + cancel rate by bin
    bins = [-1, 7, 30, 90, 180, 365, cdf["LeadTime"].max()]
    labels = ["0-7", "8-30", "31-90", "91-180", "181-365", ">365"]
    lt = pd.cut(cdf["LeadTime"], bins=bins, labels=labels)
    rate = cdf.groupby(lt, observed=True)["IsCanceled"].agg(["mean", "size"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(cdf["LeadTime"], bins=60, color=BLUE)
    ax[0].set(xlabel="LeadTime (days)", ylabel="Bookings", title="LeadTime (right-skewed)")
    ax[1].bar(rate.index.astype(str), rate["mean"], color=ORANGE)
    for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
        ax[1].text(i, m + 0.01, f"{m:.0%}\n(n={n:,})", ha="center", fontsize=8)
    ax[1].set(ylim=(0, 0.75), xlabel="LeadTime bin (days)", ylabel="Cancel rate",
              title="Cancel rate by lead time")
    _save(fig, cfg, "05_leadtime_cancel.png")

    # 6) cancel rate by categorical variables
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col in zip(ax, ["DepositType", "MarketSegment", "DistributionChannel"]):
        r = cdf.groupby(col)["IsCanceled"].agg(["mean", "size"]).sort_values("mean")
        r = r[r["size"] >= 30]
        a.barh(r.index, r["mean"], color=BLUE)
        for i, m in enumerate(r["mean"]):
            a.text(m + 0.01, i, f"{m:.0%}", va="center", fontsize=8)
        a.set(xlim=(0, 1.1), xlabel="Cancel rate", title=col)
    _save(fig, cfg, "06_cancel_by_category.png")

    # 7) correlation heatmap (numeric columns of the cleaned data)
    num = cdf.select_dtypes(include="number").drop(columns=["ArrivalDateYear"])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Correlation of numeric columns (cleaned)")
    _save(fig, cfg, "07_correlation_heatmap.png")

    # 8) bookings per month 2015-2017
    m = cdf["ArrivalDateMonth"].map({n: i + 1 for i, n in enumerate(MONTHS)})
    ym = pd.to_datetime(dict(year=cdf["ArrivalDateYear"], month=m, day=1))
    per = ym.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(per.index, per.values, "o-", color=BLUE)
    ax.set(ylabel="Bookings (cleaned)", title="Bookings per arrival month, 2015-2017")
    _save(fig, cfg, "08_bookings_per_month.png")

    # 9) leakage evidence: DepositType 'Non Refund' vs cancel
    dep = cdf.groupby("DepositType")["IsCanceled"].agg(["mean", "size"]).round(3)
    dep.to_csv(path_of(cfg, "table_dir") / "cancel_rate_by_deposit_type.csv")
    print("\nCancel rate by DepositType (leakage evidence)\n", dep.to_string())
    return cdf, report
