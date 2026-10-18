"""Load config.yaml and resolve project paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg, key):
    """Absolute path for a key under `paths:` in config.yaml (parent dirs are created)."""
    p = ROOT / cfg["paths"][key]
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p
