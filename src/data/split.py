"""Step 4: one stratified 80/20 split. TEST stays locked until the final evaluation (Milestone 2)."""
from sklearn.model_selection import train_test_split


def make_split(df, cfg):
    s = cfg["split"]
    return train_test_split(
        df,
        test_size=s["test_size"],
        random_state=cfg["seed"],
        stratify=df[s["stratify_on"]],
    )
