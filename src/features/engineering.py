"""Row-wise feature engineering (no statistics learned from data, so it is leak-free)."""


def add_engineered(df):
    df = df.copy()
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    df["TotalGuests"] = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    return df
