"""Small shared helpers: config loading, seeding, logging.

Deliberately tiny and dependency-light. Everything here is used by every
script in the project, so it should stay boring.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

# The repo root is two levels up from this file (src/utils.py -> src -> root).
# Resolving against this, rather than the current working directory, is what
# lets `python scripts/inspect.py` work from anywhere on disk.
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path="config.yaml"):
    """Read config.yaml and return it as a dict, with paths made absolute.

    Paths in the YAML are written relative to the repo root because that is
    what a human reading the file wants to see. They are converted to absolute
    strings here so that no script has to care where it was launched from.

    Returned as strings rather than Path objects so the dict stays
    JSON-serialisable - scripts/inspect.py dumps parts of it into
    results/inspect.json.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["paths"] = {
        key: str((REPO_ROOT / value).resolve())
        for key, value in config["paths"].items()
    }
    config["repo_root"] = str(REPO_ROOT)
    return config


def set_seed(seed):
    """Seed every random number generator we use, so runs repeat identically.

    torch is imported inside the function on purpose: protocol.py and the
    tests must work on a machine where torch is not installed yet, and a
    module-level `import torch` here would break that.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logging.warning("torch not installed - seeded random and numpy only")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: full GPU determinism also needs cudnn settings. That belongs in
    # phase 3 with the training loop, not here.


def setup_logging(level=logging.INFO):
    """Configure logging with timestamps. We use logging, not print."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,   # so calling this twice in one process still works
    )
