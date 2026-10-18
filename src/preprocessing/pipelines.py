"""Step 5: reusable preprocessing. Every transformer that learns from data lives INSIDE the Pipeline,
so cross-validation re-fits it on the training folds only (no leakage)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(numeric, categorical, dense=False):
    """impute (median) -> scale for numerics; one-hot (rare levels < 1% grouped) for categoricals."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=0.01), categorical),
        ],
        sparse_threshold=0 if dense else 0.3,
    )
