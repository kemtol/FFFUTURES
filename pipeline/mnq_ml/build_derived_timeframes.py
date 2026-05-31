#!/usr/bin/env python3
"""Build MNQ 5m and 15m DuckDB files from canonical MNQ 1m DuckDB.

This intentionally mirrors the MGC Level-0 shape: one file per timeframe. The
resample label uses the Super Structure convention: label="right",
closed="left".
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/Level_0_Raw/MNQ_1m.duckdb"
DEFAULT_OUT_DIR = ROOT / "data/Level_0_Raw"
SYMBOL = "MICRO_NASDAQ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_timeframe(source: Path, out_dir: Path, minutes: int, force: bool) -> dict:
    timeframe = f"{minutes}m"
    output = out_dir / f"MNQ_{timeframe}.duckdb"
    manifest_path = out_dir / f"MNQ_{timeframe}_duckdb_manifest.json"
    table = f"ohlcv_{timeframe}"

    if output.exists() and not force:
        raise SystemExit(f"Output exists; use --force to replace: {output}")
    if output.exists():
        output.unlink()

    con = duckdb.connect(str(output))
    con.execute("set preserve_insertion_order to false")
    source_sql = str(source).replace("'", "''")
    con.execute(f"attach '{source_sql}' as src (read_only)")
    interval_sql = f"interval '{minutes} minutes'"

    con.execute(
        f"""
        create table {table} as
        with base as (
          select
            time_bucket({interval_sql}, timestamp_utc) + {interval_sql} as timestamp_utc,
            timestamp_utc as source_ts,
            open,
            high,
            low,
            close,
            volume,
            source_symbol,
            date_diff(
              'second',
              lag(timestamp_utc) over (order by timestamp_utc),
              timestamp_utc
            ) as prev_gap_seconds
          from src.ohlcv_1m
        ), grouped as (
          select
            timestamp_utc,
            arg_min(open, source_ts) as open,
            max(high) as high,
            min(low) as low,
            arg_max(close, source_ts) as close,
            sum(volume)::bigint as volume,
            count(*)::int as source_bar_count,
            min(source_ts) as first_source_ts,
            max(source_ts) as last_source_ts,
            max(case when prev_gap_seconds > 60 then 1 else 0 end)::boolean as contains_source_gap,
            count(distinct source_symbol)::int as source_symbol_count
          from base
          group by timestamp_utc
        )
        select
          '{SYMBOL}'::varchar as symbol,
          '{timeframe}'::varchar as timeframe,
          epoch_ms(timestamp_utc)::bigint as epoch_ms,
          timestamp_utc,
          open::double as open,
          high::double as high,
          low::double as low,
          close::double as close,
          volume,
          source_bar_count,
          first_source_ts,
          last_source_ts,
          contains_source_gap,
          source_symbol_count
        from grouped
        order by timestamp_utc
        """
    )
    con.execute(f"create index idx_{table}_ts on {table}(timestamp_utc)")

    summary = con.execute(
        f"""
        select
          count(*)::bigint as rows,
          min(timestamp_utc)::varchar as min_ts,
          max(timestamp_utc)::varchar as max_ts,
          count(distinct timestamp_utc)::bigint as distinct_timestamps,
          sum(case when source_bar_count < {minutes} then 1 else 0 end)::bigint as partial_bars,
          sum(case when contains_source_gap then 1 else 0 end)::bigint as bars_with_source_gap,
          sum(case when high < greatest(open, close, low) then 1 else 0 end)::bigint as bad_high_rows,
          sum(case when low > least(open, close, high) then 1 else 0 end)::bigint as bad_low_rows
        from {table}
        """
    ).fetchdf().iloc[0].to_dict()
    gap_summary = con.execute(
        f"""
        with d as (
          select
            timestamp_utc,
            date_diff(
              'second',
              lag(timestamp_utc) over (order by timestamp_utc),
              timestamp_utc
            ) as gap_seconds
          from {table}
        )
        select
          count(*) filter (where gap_seconds > {minutes * 60})::bigint as gaps_gt_expected,
          max(gap_seconds)::bigint as max_gap_seconds
        from d
        """
    ).fetchdf().iloc[0].to_dict()
    con.close()

    rows = int(summary["rows"])
    distinct_timestamps = int(summary["distinct_timestamps"])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output": str(output),
        "table": table,
        "timeframe": timeframe,
        "resample_contract": {
            "label": "right",
            "closed": "left",
            "derived_from": "ohlcv_1m",
        },
        "rows": rows,
        "min_ts": summary["min_ts"],
        "max_ts": summary["max_ts"],
        "duplicate_timestamps": rows - distinct_timestamps,
        "partial_bars": int(summary["partial_bars"] or 0),
        "bars_with_source_gap": int(summary["bars_with_source_gap"] or 0),
        "bad_high_rows": int(summary["bad_high_rows"] or 0),
        "bad_low_rows": int(summary["bad_low_rows"] or 0),
        "gaps_gt_expected": int(gap_summary["gaps_gt_expected"] or 0),
        "max_gap_seconds": int(gap_summary["max_gap_seconds"] or 0),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    if not source.exists():
        raise SystemExit(f"Missing source DuckDB: {source}")

    manifests = [
        build_timeframe(source, out_dir, minutes=5, force=args.force),
        build_timeframe(source, out_dir, minutes=15, force=args.force),
    ]
    print(json.dumps(manifests, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
