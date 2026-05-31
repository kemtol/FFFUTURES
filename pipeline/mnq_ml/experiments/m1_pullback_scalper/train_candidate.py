#!/usr/bin/env python3
"""MNQ M1 trainer scaffold.

This intentionally refuses to train until the MNQ gate passes.
"""

from __future__ import annotations

import json
import subprocess
import sys

from common import load_config, project_path


def main() -> int:
    cfg = load_config()
    gate_script = "pipeline/mnq_ml/experiments/m1_pullback_scalper/gate_training_data.py"
    proc = subprocess.run([sys.executable, gate_script], cwd=project_path("."), check=False)
    if proc.returncode != 0:
        raise SystemExit("MNQ training blocked by data gate.")

    report_path = project_path(cfg["outputs"]["training_gate_report"])
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS" or report.get("training_allowed") is not True:
        raise SystemExit(f"MNQ training blocked by gate report: {report_path}")

    raise SystemExit(
        "MNQ trainer implementation is pending. Add model splits only after L1/L2 data quality is proven."
    )


if __name__ == "__main__":
    sys.exit(main())
