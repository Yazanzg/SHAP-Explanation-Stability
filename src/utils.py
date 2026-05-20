"""Shared utilities: logging, directory creation, typing helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once for CLI and notebooks."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def ensure_directories(*paths: Path) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
