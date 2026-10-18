"""Step 1: load H2.csv and verify it is the City Hotel file (79,330 x 31)."""
import pandas as pd

from src.config import path_of


def strip_text(df):
    """The CSV pads some text values with spaces (e.g. '      NULL'); remove them."""
    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    return df


def load_raw(cfg, verify=True):
    df = pd.read_csv(path_of(cfg, "raw"))
    if verify:
        expected = tuple(cfg["expected_shape"])
        assert df.shape == expected, f"H2.csv shape {df.shape} != expected {expected}"
    return df
