#!/usr/bin/env python3
"""Build MNQ M1 pullback event datamart.

The event builder is intentionally blocked until L1 feature engineering is
implemented and gated. It documents the executable timing contract for later
implementation.
"""

from __future__ import annotations

import argparse
import sys

from common import assert_mnq_namespaces, load_config, project_path


REQUIRED_L2_COLUMNS = [
    "event_id",
    "signal_ts",
    "entry_ts",
    "side",
    "signal_open",
    "signal_high",
    "signal_low",
    "signal_close",
    "signal_volume",
    "entry_open",
    "entry_high",
    "entry_low",
    "entry_close",
    "entry_volume",
    "entry_gap_seconds",
    "signal_prev_gap_seconds",
    "signal_data_quality_ok",
    "entry_price",
    "sl_price",
    "tp_price",
    "exit_ts",
    "exit_price",
    "exit_reason",
    "hold_bars",
    "risk_pts",
    "label",
    "pnl_usd",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    assert_mnq_namespaces(cfg)
    l1_path = project_path(cfg["outputs"]["l1_context"])

    print("MNQ M1 event builder scaffold")
    print(f"l1_exists={l1_path.exists()} l1={l1_path}")
    print(f"output={cfg['outputs']['events']}")
    print("timing=signal on M1 close, entry at next M1 open")

    if args.dry_run:
        return 0

    if not l1_path.exists():
        raise SystemExit(f"Missing MNQ L1 context: {l1_path}")
    raise SystemExit("L2 event implementation is pending after MNQ L1 feature builder is complete.")


if __name__ == "__main__":
    sys.exit(main())
