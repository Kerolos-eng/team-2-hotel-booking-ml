"""CLI for the CSE382 City Hotel (H2) project.   Milestone 1 commands:

    python main.py analyze                  # load + verify H2.csv, EDA tables + figures
    python main.py clean                    # cleaning -> data/H2_clean.csv
    python main.py split                    # stratified 80/20 -> data/train.csv, data/test.csv
    python main.py baseline --task all      # one untuned baseline per task (CV on TRAIN only)
    python main.py all                      # everything above, in order

Milestone 2 will add: train | tune | evaluate | cluster | predict (loading saved artifacts).
"""
import argparse

from src.config import load_config, path_of


def cmd_analyze(cfg):
    from src.data import eda
    eda.run(cfg)


def cmd_clean(cfg):
    from src.data.cleaning import clean
    from src.data.loader import load_raw
    df, report = clean(load_raw(cfg), cfg["cleaning"]["adr_max"])
    df.to_csv(path_of(cfg, "clean"), index=False)
    for k, v in report.items():
        print(f"{k}: {v}")


def cmd_split(cfg):
    import pandas as pd
    from src.data.split import make_split
    train, test = make_split(pd.read_csv(path_of(cfg, "clean")), cfg)
    train.to_csv(path_of(cfg, "train"), index=False)
    test.to_csv(path_of(cfg, "test"), index=False)
    print(f"train: {train.shape}  test: {test.shape}")
    print(f"cancel rate  train: {train['IsCanceled'].mean():.4f}  test: {test['IsCanceled'].mean():.4f}")


def cmd_baseline(cfg, task):
    if task in ("regression", "all"):
        print("\n=== Task A: regression (ADR) ===")
        from src.regression import baseline
        baseline.run(cfg)
    if task in ("classification", "all"):
        print("\n=== Task B: classification (IsCanceled) ===")
        from src.classification import baseline
        baseline.run(cfg)
    if task in ("clustering", "all"):
        print("\n=== Task C: clustering ===")
        from src.clustering import baseline
        baseline.run(cfg)


def main():
    parser = argparse.ArgumentParser(description="CSE382 City Hotel (H2) project")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("analyze")
    sub.add_parser("clean")
    sub.add_parser("split")
    b = sub.add_parser("baseline")
    b.add_argument("--task", choices=["regression", "classification", "clustering", "all"], default="all")
    sub.add_parser("all")
    args = parser.parse_args()
    cfg = load_config()

    if args.command == "analyze":
        cmd_analyze(cfg)
    elif args.command == "clean":
        cmd_clean(cfg)
    elif args.command == "split":
        cmd_split(cfg)
    elif args.command == "baseline":
        cmd_baseline(cfg, args.task)
    elif args.command == "all":
        cmd_analyze(cfg)
        cmd_clean(cfg)
        cmd_split(cfg)
        cmd_baseline(cfg, "all")


if __name__ == "__main__":
    main()
