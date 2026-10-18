"""Task B (Milestone 1): baseline classification for IsCanceled + two leakage ablations.

Main model uses only booking-time features. Ablations (default rule for 'doubt'/'leak' columns):
  * + DepositType        -> 'doubt' column, report the gap
  * + ReservationStatus  -> 'leak' column, on purpose, to SHOW what leakage looks like (never deployed)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import path_of
from src.evaluation.cv import run_cv, save_table
from src.features.feature_sets import FEATURES
from src.preprocessing.pipelines import make_preprocessor

SCORING = {  # precision uses zero_division=0 so the Dummy model (predicts no positives) does not warn
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
}


def run(cfg):
    train = pd.read_csv(path_of(cfg, "train"))
    y = train["IsCanceled"]
    num, cat = FEATURES["classification"]["numeric"], FEATURES["classification"]["categorical"]
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])
    logreg = lambda: LogisticRegression(max_iter=1000, random_state=cfg["seed"])  # noqa: E731

    rows = [
        run_cv("Dummy (prior)", DummyClassifier(strategy="prior"), train[num + cat], y, num, cat, cv, SCORING,
               extra={"task": "classification", "config": "baseline"}),
        run_cv("LogReg", logreg(), train[num + cat], y, num, cat, cv, SCORING,
               extra={"task": "classification", "config": "baseline"}),
        run_cv("LogReg + DepositType", logreg(), train[num + cat + ["DepositType"]], y, num,
               cat + ["DepositType"], cv, SCORING,
               extra={"task": "classification", "config": "ablation (doubt column)"}),
        run_cv("LogReg + ReservationStatus", logreg(), train[num + cat + ["ReservationStatus"]], y, num,
               cat + ["ReservationStatus"], cv, SCORING,
               extra={"task": "classification", "config": "LEAKAGE demo (never deployed)"}),
    ]
    table = pd.DataFrame(rows)
    save_table(table, cfg, "baseline_classification")
    cols = ["model", "config", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean",
            "average_precision_mean", "roc_auc_mean"]
    print(table[cols].round(3).to_string(index=False))

    # normalised confusion matrix of the baseline LogReg (out-of-fold predictions)
    pipe = Pipeline([("prep", make_preprocessor(num, cat)), ("model", logreg())])
    pred = cross_val_predict(pipe, train[num + cat], y, cv=cv)
    cm = confusion_matrix(y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not canceled", "Canceled"],
           yticklabels=["Not canceled", "Canceled"], xlabel="Predicted", ylabel="Actual",
           title="LogReg baseline (row-normalised)")
    fig.tight_layout()
    fig.savefig(path_of(cfg, "fig_dir") / "clf_confusion_matrix.png", dpi=150)
    plt.close(fig)
    return table
