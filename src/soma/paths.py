"""Packaged defaults and local workspace paths."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def workspace_root():
    if os.environ.get("SOMA_WORKSPACE"):
        return Path(os.environ["SOMA_WORKSPACE"]).expanduser().resolve()
    for parent in PACKAGE_ROOT.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "soma").is_dir():
            return parent
    return Path.cwd()


ROOT = workspace_root()
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "main.json"
DEFAULT_HEAD_CONFIG = PACKAGE_ROOT / "configs" / "augmentation.json"
DEFAULT_DATA_DIR = ROOT / "data" / "private"
