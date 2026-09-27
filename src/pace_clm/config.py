"""Versioned configuration loading with basic contract validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return data


def load_mvp_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = load_yaml(path)
    required = {"version", "source", "model", "streaming", "projection", "labels"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    actions_path = path.parent / str(config["actions_config"])
    config["actions"] = load_yaml(actions_path)
    return config
