#!/usr/bin/env python3
"""Shared helpers for MNQ ORB volatility-targeted research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def project_path(path_value: str) -> Path:
    return ROOT / path_value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def assert_mnq_namespaces(cfg: dict[str, Any]) -> None:
    outputs = cfg["outputs"]
    required_prefixes = {
        "l1_context": "data/Level_1_Features/mnq/",
        "l1_manifest": "data/Level_1_Features/mnq/",
        "events": "data/Level_2_Datamart/mnq/",
        "events_manifest": "data/Level_2_Datamart/mnq/",
        "training_gate_report": "data/Level_2_Datamart/mnq/",
        "model_dir": "model/MNQ/",
    }
    bad = {
        key: value
        for key, prefix in required_prefixes.items()
        for value in [outputs.get(key, "")]
        if not value.startswith(prefix)
    }
    if bad:
        raise SystemExit(f"MNQ namespace violation in config outputs: {bad}")
