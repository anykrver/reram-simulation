"""Load and validate YAML configs for experiments and CLI."""
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file.

    Args:
        path: Path to .yaml file (relative to cwd or absolute).

    Returns:
        Nested dict of config. Empty dict if file is empty.

    Raises:
        FileNotFoundError: If path does not exist.
        yaml.YAMLError: If YAML is invalid.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def get_crossbar_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract crossbar config with defaults. Safe for missing keys."""
    cb = cfg.get("crossbar") or {}
    return {
        "rows": int(cb.get("rows", 64)),
        "cols": int(cb.get("cols", 64)),
        "precision": cb.get("precision", "float32"),
    }


def get_seed(cfg: Dict[str, Any]) -> int:
    """Extract seed with default 42."""
    return int(cfg.get("seed", 42))
