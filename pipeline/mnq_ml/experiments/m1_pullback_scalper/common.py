#!/usr/bin/env python3
"""Shared helpers for the MNQ M1 pullback scaffold."""

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


def source_status(cfg: dict[str, Any]) -> dict[str, Any]:
    source = cfg["source"]
    source_path = project_path(source["db"])
    return {
        "ready": bool(source.get("ready", False)),
        "path": str(source_path),
        "exists": source_path.exists(),
        "table": source.get("table"),
        "required_columns": source.get("required_columns", []),
    }


def require_source_ready(cfg: dict[str, Any]) -> None:
    status = source_status(cfg)
    if not status["ready"]:
        raise SystemExit(
            "MNQ source is not ready. Verify L0 data, then set source.ready=true in config.json."
        )
    if not status["exists"]:
        raise SystemExit(f"MNQ source DB does not exist: {status['path']}")
