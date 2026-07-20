"""
src/utils/io.py

Small, reusable I/O utilities for configuration loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file and return a dict.

    Args:
        path: Path to the YAML config file.

    Returns:
        A dictionary with the YAML contents.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if YAML cannot be parsed into a dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping/dict at top-level: {p}")
    return data
