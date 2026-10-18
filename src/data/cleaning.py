"""Step 2: cleaning. Only row-level rules that do not learn anything from the data.

Anything data-dependent (imputer statistics, scaler, encoder, PCA ...) lives inside the
sklearn Pipeline and is fitted on TRAIN folds only.
"""
import pandas as pd

from src.data.loader import strip_text


def clean(df, adr_max=1000):
    """Return (clean_df, report). Order: strip -> drop duplicates -> drop invalid rows -> fill NaN."""
    df = strip_text(df)
    report = {"raw_rows": len(df), "raw_cancel_rate": df["IsCanceled"].mean()}

    # 1) exact duplicate rows (removed BEFORE the split so no identical row lands in train and test)
    n = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = n - len(df)
    report["cancel_rate_after_dedup"] = df["IsCanceled"].mean()

    # 2) invalid bookings (counts may overlap; a row is dropped if ANY rule fires)
    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    rules = {
        "zero_guests": guests == 0,
        "zero_nights": nights == 0,
        "adr_negative": df["ADR"] < 0,
        "adr_outlier": df["ADR"] >= adr_max,
    }
    for name, mask in rules.items():
        report[f"invalid_{name}"] = int(mask.sum())
    invalid = pd.concat(rules, axis=1).any(axis=1)
    report["invalid_rows_removed"] = int(invalid.sum())
    df = df[~invalid].copy()

    # 3) missing values with a documented meaning
    report["children_nan_filled"] = int(df["Children"].isna().sum())
    report["country_nan_filled"] = int(df["Country"].isna().sum())
    df["Children"] = df["Children"].fillna(0)
    df["Country"] = df["Country"].fillna("Unknown")
    # Agent / Company == "NULL" means "no agent / no company" (not a missing value) -> flags
    df["HasAgent"] = (df["Agent"].notna() & (df["Agent"] != "NULL")).astype(int)
    df["HasCompany"] = (df["Company"].notna() & (df["Company"] != "NULL")).astype(int)

    report["clean_rows"] = len(df)
    report["clean_cancel_rate"] = df["IsCanceled"].mean()
    report["total_removed"] = report["raw_rows"] - report["clean_rows"]
    return df.reset_index(drop=True), report
