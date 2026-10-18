"""Task A (Milestone 1): baseline regression for ADR. Dummy floor + Ridge, 5-fold CV on TRAIN only."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import path_of
from src.evaluation.cv import run_cv, save_table
from src.features.engineering import add_engineered
from src.features.feature_sets import FEATURES
from src.preprocessing.pipelines import make_preprocessor

SCORING = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
SIGN = {"neg_mean_absolute_error": -1, "neg_root_mean_squared_error": -1}


def run(cfg):
    train = add_engineered(pd.read_csv(path_of(cfg, "train")))
    num, cat = FEATURES["regression"]["numeric"], FEATURES["regression"]["categorical"]
    X, y = train[num + cat], train["ADR"]
    cv = KFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])

    models = {"Dummy (mean)": DummyRegressor(strategy="mean"), "Ridge (alpha=1)": Ridge(alpha=1.0)}
    rows = [run_cv(n, m, X, y, num, cat, cv, SCORING, SIGN, {"task": "regression", "config": "baseline"})
            for n, m in models.items()]
    table = pd.DataFrame(rows).rename(columns={
        "neg_mean_absolute_error_mean": "MAE_mean", "neg_mean_absolute_error_std": "MAE_std",
        "neg_root_mean_squared_error_mean": "RMSE_mean", "neg_root_mean_squared_error_std": "RMSE_std",
        "r2_mean": "R2_mean", "r2_std": "R2_std"})
    save_table(table, cfg, "baseline_regression")
    print(table[["model", "MAE_mean", "RMSE_mean", "R2_mean"]].round(3).to_string(index=False))

    # predicted vs actual (out-of-fold predictions, so every point was predicted by a model that never saw it)
    pipe = Pipeline([("prep", make_preprocessor(num, cat)), ("model", Ridge(alpha=1.0))])
    pred = cross_val_predict(pipe, X, y, cv=cv)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].scatter(y, pred, s=3, alpha=0.25, color="#2b5d9b")
    lim = [0, max(y.max(), pred.max())]
    ax[0].plot(lim, lim, color="#c0392b", lw=1)
    ax[0].set(xlabel="Actual ADR", ylabel="Predicted ADR (out-of-fold)", title="Ridge: predicted vs actual")
    ax[1].hist(y - pred, bins=60, color="#2b5d9b")
    ax[1].axvline(0, color="#c0392b", lw=1)
    ax[1].set(xlabel="Residual (actual - predicted)", ylabel="Bookings", title="Residuals")
    fig.tight_layout()
    fig.savefig(path_of(cfg, "fig_dir") / "reg_pred_vs_actual.png", dpi=150)
    plt.close(fig)
    return table
