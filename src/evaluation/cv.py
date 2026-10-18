"""Small helpers to run cross-validation and tabulate mean +/- std."""
import time

import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing.pipelines import make_preprocessor


def run_cv(name, model, X, y, numeric, categorical, cv, scoring, sign=None, extra=None):
    """Cross-validate Pipeline(preprocessor -> model). `sign` flips neg_* scorers to positive errors."""
    pipe = Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", model)])
    t0 = time.time()
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    row = {"model": name, **(extra or {})}
    for metric in scoring:
        vals = res[f"test_{metric}"]
        vals = -vals if (sign or {}).get(metric) == -1 else vals
        row[f"{metric}_mean"] = vals.mean()
        row[f"{metric}_std"] = vals.std()
    row["cv_seconds"] = time.time() - t0
    return row


def save_table(df, cfg, name):
    from src.config import path_of
    out = path_of(cfg, "table_dir") / f"{name}.csv"
    df.to_csv(out, index=False)
    return out
