#!/usr/bin/env python3
"""Build M1 ORB context for MNQ New York sessions."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from common import assert_mnq_namespaces, load_config, project_path, write_json

REQUIRED_L1_COLUMNS = [
    "timestamp_utc",
    "ny_date",
    "ny_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source_bar_count",
    "contains_source_gap",
    "bar_data_quality_ok",
    "orb_high",
    "orb_low",
    "orb_range_pts",
    "orb_bar_count",
    "orb_complete",
    "minutes_from_open",
    "eligible_after_or",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser.parse_args()


def load_source(cfg: dict) -> pd.DataFrame:
    source = cfg["source"]
    db_path = project_path(source["db"])
    if not db_path.exists():
        raise SystemExit(f"Missing MNQ source DB: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        f"""
        with base as (
          select
            timestamp_utc,
            open,
            high,
            low,
            close,
            volume,
            date_diff(
              'second',
              lag(timestamp_utc) over (order by timestamp_utc),
              timestamp_utc
            ) as prev_gap_seconds
          from {source["table"]}
        )
        select
          timestamp_utc + interval '1 minute' as timestamp_utc,
          open,
          high,
          low,
          close,
          volume,
          1::int as source_bar_count,
          coalesce(prev_gap_seconds > 60, false)::boolean as contains_source_gap
        from base
        order by timestamp_utc
        """
    ).fetchdf()
    con.close()
    return df


def build_context(df: pd.DataFrame, cfg: dict, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    tz = ZoneInfo(cfg["session"]["timezone"])
    open_time = cfg["session"]["market_open"]
    orb_end = cfg["session"]["orb_end"]
    exit_time = cfg["session"]["time_exit"]
    tf_minutes = int(cfg["session"]["decision_timeframe_minutes"])

    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    if start_date:
        out = out[out["timestamp_utc"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        out = out[out["timestamp_utc"] < pd.Timestamp(end_date, tz="UTC")]

    ny = out["timestamp_utc"].dt.tz_convert(tz)
    out["ny_date"] = ny.dt.date.astype(str)
    out["ny_time"] = ny.dt.strftime("%H:%M")
    open_minutes = int(open_time[:2]) * 60 + int(open_time[3:])
    bar_minutes = ny.dt.hour * 60 + ny.dt.minute
    out["minutes_from_open"] = (bar_minutes - open_minutes).astype(int)
    valid_ohlc = (
        out[["open", "high", "low", "close"]].notna().all(axis=1)
        & (out["high"] >= out[["open", "close", "low"]].max(axis=1))
        & (out["low"] <= out[["open", "close", "high"]].min(axis=1))
        & (out["volume"].fillna(-1) >= 0)
    )
    out["bar_data_quality_ok"] = (
        valid_ohlc
        & (out["source_bar_count"].astype(int) == tf_minutes)
        & (~out["contains_source_gap"].astype(bool))
    )

    orb_mask = (
        (out["ny_time"] > open_time)
        & (out["ny_time"] <= orb_end)
        & out["bar_data_quality_ok"]
    )
    orb = (
        out.loc[orb_mask]
        .groupby("ny_date", as_index=False)
        .agg(
            orb_high=("high", "max"),
            orb_low=("low", "min"),
            orb_bar_count=("timestamp_utc", "count"),
        )
    )
    orb["orb_range_pts"] = orb["orb_high"] - orb["orb_low"]
    expected_orb_bars = 30 // tf_minutes
    orb["orb_complete"] = orb["orb_bar_count"] == expected_orb_bars
    out = out.merge(orb, on="ny_date", how="left")
    day_orb_complete = out["orb_complete"].eq(True)
    out["orb_bar_count"] = out["orb_bar_count"].fillna(0).astype(int)
    post_or_mask = out["ny_time"] > orb_end
    pre_or_mask = ~post_or_mask
    out.loc[pre_or_mask, ["orb_high", "orb_low", "orb_range_pts"]] = pd.NA
    out.loc[pre_or_mask, "orb_bar_count"] = 0
    out["orb_complete"] = day_orb_complete & post_or_mask
    out["eligible_after_or"] = (
        out["bar_data_quality_ok"]
        & out["orb_complete"]
        & (out["ny_time"] <= exit_time)
    )
    return out.reset_index(drop=True)


def validate_l1(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_L1_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing L1 columns: {missing}")
    hard = ["timestamp_utc", "ny_date", "ny_time", "open", "high", "low", "close", "volume"]
    nulls = df[hard].isna().sum()
    bad = nulls[nulls > 0]
    if not bad.empty:
        raise SystemExit(f"L1 hard-null failure: {bad.to_dict()}")


def main() -> int:
    args = parse_args()
    cfg = load_config()
    assert_mnq_namespaces(cfg)
    out_path = project_path(cfg["outputs"]["l1_context"])
    manifest_path = project_path(cfg["outputs"]["l1_manifest"])
    if out_path.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"Output exists; use --force: {out_path}")

    src = load_source(cfg)
    l1 = build_context(src, cfg, args.start_date, args.end_date)
    validate_l1(l1)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(l1)),
        "min_ts": l1["timestamp_utc"].min().isoformat() if not l1.empty else None,
        "max_ts": l1["timestamp_utc"].max().isoformat() if not l1.empty else None,
        "ny_days": int(l1["ny_date"].nunique()),
        "orb_complete_days": int(l1.loc[l1["orb_complete"], "ny_date"].nunique()),
        "eligible_after_or_bars": int(l1["eligible_after_or"].sum()),
        "bar_quality_ok_rows": int(l1["bar_data_quality_ok"].sum()),
        "output": str(out_path),
    }
    print(manifest)
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        l1.to_parquet(out_path, index=False)
        write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
