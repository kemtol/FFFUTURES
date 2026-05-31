#!/usr/bin/env python3
"""Build MNQ M1 L1 context.

This is a scaffold guard, not a completed feature builder. It intentionally
refuses to run until the MNQ L0 source is marked ready in config.json.
"""

from __future__ import annotations

import argparse
import sys

from common import assert_mnq_namespaces, load_config, require_source_ready, source_status


REQUIRED_L1_COLUMNS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "prev_gap_seconds",
    "data_quality_ok",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    assert_mnq_namespaces(cfg)
    status = source_status(cfg)

    print("MNQ L1 context builder scaffold")
    print(f"source_ready={status['ready']} source_exists={status['exists']} source={status['path']}")
    print(f"output={cfg['outputs']['l1_context']}")

    if args.dry_run:
        return 0

    require_source_ready(cfg)
    raise SystemExit(
        "L1 feature implementation is pending. Add the MNQ SQLite/Databento adapter here after L0 schema is verified."
    )


if __name__ == "__main__":
    sys.exit(main())
