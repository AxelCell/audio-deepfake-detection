"""Config loading, seeding and logging."""

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

# src/utils.py -> src -> repo root. Resolving against this instead of the
# working directory is what lets the scripts run from any folder.
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path="config.yaml"):
    """Read config.yaml and return it as a dict, with paths made absolute."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Left as strings rather than Path objects so inspect.py can write them
    # straight into results/inspect.json.
    config["paths"] = {
        key: str((REPO_ROOT / value).resolve())
        for key, value in config["paths"].items()
    }
    config["repo_root"] = str(REPO_ROOT)
    return config


def repo_relative(path, config):
    """Path relative to the repo root, with forward slashes.

    Anything written into a committed artifact goes through this. An absolute
    path bakes in one person's username and folder layout, so the same run on
    two machines would produce artifacts that differ for no real reason.

    Falls back to the full path if it is outside the repo, which is what the
    tests do when they point the config at a temp folder.
    """
    try:
        return Path(path).resolve().relative_to(config["repo_root"]).as_posix()
    except ValueError:
        return str(path)


def set_seed(seed):
    """Seed random, numpy and torch so two runs give the same answer."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch is imported here, not at the top of the file, so that the tests
    # still run on a machine where it isn't installed yet.
    try:
        import torch
    except ImportError:
        logging.warning("torch not installed - seeded random and numpy only")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
