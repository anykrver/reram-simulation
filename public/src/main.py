"""CLI: load config, run ideal vs non-ideal crossbar, log metrics."""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root on path when run as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.crossbar import IdealCrossbar, NonIdealCrossbar
from src.utils.config_loader import load_config, get_crossbar_config, get_seed
from src.utils.logger import get_logger
from src.hardware.energy_estimator import EnergyEstimator

LOG = get_logger(__name__)


def run_ideal(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run ideal crossbar: I = V @ G, then compute energy.

    Args:
        cfg: Config dict with keys crossbar (rows, cols), seed.

    Returns:
        Dict with I_shape and energy_J.
    """
    cb = get_crossbar_config(cfg)
    rows, cols = cb["rows"], cb["cols"]
    seed = get_seed(cfg)
    np.random.seed(seed)
    crossbar = IdealCrossbar(rows, cols)
    G = np.random.rand(rows, cols).astype(np.float32) * 0.1
    crossbar.set_conductance(G)
    V = np.random.rand(rows).astype(np.float32) * 0.5
    I = crossbar.run(V)
    LOG.info("Ideal crossbar: I shape %s", I.shape)
    est = EnergyEstimator()
    E = est.energy_crossbar(V, G)
    LOG.info("Energy (J): %.6e", E)
    return {"I_shape": I.shape, "energy_J": float(E)}


def run_non_ideal(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run non-ideal crossbar with optional noise, IR drop, variability.

    Args:
        cfg: Config with crossbar, noise, ir_drop, variability, seed.

    Returns:
        Dict with I_shape.
    """
    cb = get_crossbar_config(cfg)
    rows, cols = cb["rows"], cb["cols"]
    seed = get_seed(cfg)
    noise = cfg.get("noise") or {}
    ir_drop = cfg.get("ir_drop") or {}
    var = cfg.get("variability") or {}
    crossbar = NonIdealCrossbar(
        rows,
        cols,
        noise_std=float(noise.get("std", 0)) if noise.get("enabled") else 0.0,
        ir_drop_row_r=float(ir_drop.get("row_resistance", 0)) if ir_drop.get("enabled") else 0.0,
        ir_drop_col_r=float(ir_drop.get("col_resistance", 0)) if ir_drop.get("enabled") else 0.0,
        variability_std=float(var.get("conductance_std", 0)) if var.get("enabled") else 0.0,
        seed=seed,
    )
    np.random.seed(seed)
    G = np.random.rand(rows, cols).astype(np.float32) * 0.1
    crossbar.set_conductance(G)
    V = np.random.rand(rows).astype(np.float32) * 0.5
    I = crossbar.run(V)
    LOG.info("Non-ideal crossbar: I shape %s", I.shape)
    return {"I_shape": I.shape}


def main() -> int:
    """Entry point: parse args, load config, run selected mode. Returns exit code."""
    parser = argparse.ArgumentParser(
        description="Neuro-Edge ReRAM Simulator CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ideal.yaml",
        help="Path to config YAML (relative to project root or absolute)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ideal", "non_ideal", "both"],
        default="ideal",
        help="Run ideal, non_ideal, or both",
    )
    args = parser.parse_args()
    config_path = _ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.exists():
        LOG.error("Config not found: %s", config_path)
        return 1
    try:
        cfg = load_config(str(config_path))
    except Exception as e:
        LOG.exception("Failed to load config: %s", e)
        return 1
    try:
        if args.mode in ("ideal", "both"):
            run_ideal(cfg)
        if args.mode in ("non_ideal", "both"):
            run_non_ideal(cfg)
    except Exception as e:
        LOG.exception("Run failed: %s", e)
        return 1
    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
