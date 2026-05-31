#!/usr/bin/env python3
"""Audit MNQ DuckDB L0 continuity and basic OHLCV integrity."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data/Level_0_Raw/MNQ_1m.duckdb"
DEFAULT_REPORT = ROOT / "data/Level_0_Raw/MNQ_1m_continuity_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def fetch_dict(con: duckdb.DuckDBPyConnection, query: str) -> dict:
    df = con.execute(query).fetchdf()
    return df.iloc[0].to_dict() if not df.empty else {}


def fetch_records(con: duckdb.DuckDBPyConnection, query: str) -> list[dict]:
    return con.execute(query).fetchdf().to_dict(orient="records")


def normalize(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    report_path = Path(args.report)
    if not db_path.exists():
        raise SystemExit(f"Missing MNQ DuckDB: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)
    base = fetch_dict(
        con,
        """
        select
          count(*)::bigint as rows,
          min(timestamp_utc)::varchar as min_ts,
          max(timestamp_utc)::varchar as max_ts,
          count(distinct timestamp_utc)::bigint as distinct_timestamps,
          count(distinct source_symbol)::bigint as source_symbols,
          sum(case when timestamp_utc is null then 1 else 0 end)::bigint as null_ts,
          sum(case when open is null or high is null or low is null or close is null or volume is null then 1 else 0 end)::bigint as null_ohlcv,
          sum(case when high < greatest(open, close, low) then 1 else 0 end)::bigint as bad_high_rows,
          sum(case when low > least(open, close, high) then 1 else 0 end)::bigint as bad_low_rows,
          sum(case when volume < 0 then 1 else 0 end)::bigint as negative_volume_rows
        from ohlcv_1m
        """,
    )
    gap_summary = fetch_dict(
        con,
        """
        with d as (
          select
            timestamp_utc,
            lag(timestamp_utc) over (order by timestamp_utc) as prev_ts,
            date_diff('second', lag(timestamp_utc) over (order by timestamp_utc), timestamp_utc) as gap_seconds
          from ohlcv_1m
        ), g as (
          select * from d where gap_seconds > 60
        )
        select
          count(*)::bigint as gap_count_gt_60s,
          min(gap_seconds)::bigint as min_gap_seconds,
          approx_quantile(gap_seconds, 0.50)::bigint as p50_gap_seconds,
          approx_quantile(gap_seconds, 0.90)::bigint as p90_gap_seconds,
          approx_quantile(gap_seconds, 0.99)::bigint as p99_gap_seconds,
          max(gap_seconds)::bigint as max_gap_seconds,
          sum(case when gap_seconds <= 300 then 1 else 0 end)::bigint as gaps_1m_to_5m,
          sum(case when gap_seconds > 300 and gap_seconds <= 3600 then 1 else 0 end)::bigint as gaps_5m_to_1h,
          sum(case when gap_seconds > 3600 and gap_seconds <= 7200 then 1 else 0 end)::bigint as gaps_1h_to_2h,
          sum(case when gap_seconds > 7200 then 1 else 0 end)::bigint as gaps_gt_2h
        from g
        """,
    )
    gaps_by_year = fetch_records(
        con,
        """
        with d as (
          select
            timestamp_utc,
            date_diff('second', lag(timestamp_utc) over (order by timestamp_utc), timestamp_utc) as gap_seconds
          from ohlcv_1m
        ), g as (
          select * from d where gap_seconds > 60
        )
        select
          year(timestamp_utc)::int as year,
          count(*)::bigint as gap_count,
          sum(case when gap_seconds <= 300 then 1 else 0 end)::bigint as gaps_1m_to_5m,
          sum(case when gap_seconds > 300 and gap_seconds <= 3600 then 1 else 0 end)::bigint as gaps_5m_to_1h,
          sum(case when gap_seconds > 3600 and gap_seconds <= 7200 then 1 else 0 end)::bigint as gaps_1h_to_2h,
          sum(case when gap_seconds > 7200 then 1 else 0 end)::bigint as gaps_gt_2h,
          max(gap_seconds)::bigint as max_gap_seconds
        from g
        group by 1
        order by 1
        """,
    )
    top_gaps = fetch_records(
        con,
        """
        with d as (
          select
            timestamp_utc,
            lag(timestamp_utc) over (order by timestamp_utc) as prev_ts,
            date_diff('second', lag(timestamp_utc) over (order by timestamp_utc), timestamp_utc) as gap_seconds
          from ohlcv_1m
        )
        select prev_ts::varchar as prev_ts, timestamp_utc::varchar as timestamp_utc, gap_seconds::bigint as gap_seconds
        from d
        where gap_seconds > 60
        order by gap_seconds desc
        limit 20
        """,
    )
    short_gaps_after_2022 = fetch_records(
        con,
        """
        with d as (
          select
            timestamp_utc,
            lag(timestamp_utc) over (order by timestamp_utc) as prev_ts,
            date_diff('second', lag(timestamp_utc) over (order by timestamp_utc), timestamp_utc) as gap_seconds
          from ohlcv_1m
        )
        select prev_ts::varchar as prev_ts, timestamp_utc::varchar as timestamp_utc, gap_seconds::bigint as gap_seconds
        from d
        where timestamp_utc >= timestamp '2022-01-01'
          and gap_seconds > 60
          and gap_seconds <= 3600
        order by gap_seconds desc, timestamp_utc
        limit 50
        """,
    )
    con.close()

    duplicate_timestamps = int(base["rows"] - base["distinct_timestamps"])
    hard_integrity_pass = (
        duplicate_timestamps == 0
        and int(base["null_ts"]) == 0
        and int(base["null_ohlcv"]) == 0
        and int(base["bad_high_rows"]) == 0
        and int(base["bad_low_rows"]) == 0
        and int(base["negative_volume_rows"]) == 0
    )
    continuity_status = "PASS_WITH_GAPS_REQUIRING_L1_QUARANTINE"
    if int(gap_summary.get("gaps_1m_to_5m") or 0) == 0 and int(gap_summary.get("gaps_5m_to_1h") or 0) == 0:
        continuity_status = "PASS_SCHEDULED_GAPS_ONLY_LIKELY"

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db": str(db_path),
        "hard_integrity_pass": hard_integrity_pass,
        "continuity_status": continuity_status,
        "base": {k: normalize(v) for k, v in base.items()},
        "duplicate_timestamps": duplicate_timestamps,
        "gap_summary": {k: normalize(v) for k, v in gap_summary.items()},
        "gaps_by_year": [{k: normalize(v) for k, v in row.items()} for row in gaps_by_year],
        "top_gaps": top_gaps,
        "short_gaps_after_2022": short_gaps_after_2022,
        "training_note": (
            "L1 builder must compute prev_gap_seconds and quarantine bars after any gap >60s. "
            "Do not train across gap windows."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if hard_integrity_pass else 1


if __name__ == "__main__":
    sys.exit(main())
