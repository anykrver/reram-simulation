"""Centralized logging (logging + optional file handler)."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "neuro_edge_reram",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name: Logger name.
        level: Logging level (default INFO).
        log_file: If set, also log to this file.

    Returns:
        Configured Logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
