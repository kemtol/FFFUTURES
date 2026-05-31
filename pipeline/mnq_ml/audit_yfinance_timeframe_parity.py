#!/usr/bin/env python3
"""Parity-check MNQ derived 5m/15m DuckDB bars against Yahoo direct bars.

Yahoo labels intraday bars on the left edge. The MNQ pipeline labels derived
bars on the right edge (`label="right", closed="left"`), so this audit shifts
Yahoo timestamps by the interval before comparing OHLCV.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "data/Level_0_Raw/MNQ_yfinance_timeframe_parity_report.json"
DEFAULT_SOURCE_1M = ROOT / "data/Level_0_Raw/MNQ_1m.duckdb"
TICKER = "MNQ=F"
SOURCE_SYMBOL = "MNQ=F_YF"
TF_SPECS = {
    "5m": {
        "minutes": 5,
        "db": ROOT / "data/Level_0_Raw/MNQ_5m.duckdb",
        "table": "ohlcv_5m",
    },
    "15m": {
        "minutes": 15,
        "db": ROOT / "data/Level_0_Raw/MNQ_15m.duckdb",
        "table": "ohlcv_15m",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="8d")
    parser.add_argument("--source-1m-db", default=str(DEFAULT_SOURCE_1M))
    parser.add_argument("--scope", choices=["all-overlap", "yfinance-segment"], default="yfinance-segment")
    parser.add_argument("--intervals", nargs="+", default=["5m", "15m"], choices=sorted(TF_SPECS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-latest-bars", type=int, default=3)
    parser.add_argument("--price-tolerance", type=float, default=1e-9)
    parser.add_argument("--volume-tolerance", type=float, default=0.0)
    parser.add_argument("--max-mismatch-rate", type=float, default=0.0)
    return parser.parse_args()


def normalize_yfinance(raw: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df["timestamp_utc"] = (
        pd.to_datetime(df[ts_col], utc=True)
        .dt.tz_convert(None)
        + pd.Timedelta(minutes=minutes)
    )
    out = df.rename(
        columns={
            "Open": "yf_open",
            "High": "yf_high",
            "Low": "yf_low",
            "Close": "yf_close",
            "Volume": "yf_volume",
        }
    )[["timestamp_utc", "yf_open", "yf_high", "yf_low", "yf_close", "yf_volume"]]
    out["yf_volume"] = out["yf_volume"].fillna(0).astype("int64")
    return out.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values("timestamp_utc")


def yfinance_source_bounds(source_1m_db: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    if not source_1m_db.exists():
        return None, None, 0
    con = duckdb.connect(str(source_1m_db), read_only=True)
    row = con.execute(
        """
        select min(timestamp_utc), max(timestamp_utc), count(*)::bigint
        from ohlcv_1m
        where source_symbol = ?
        """,
        [SOURCE_SYMBOL],
    ).fetchone()
    con.close()
    if row[2] == 0:
        return None, None, 0
    return pd.Timestamp(row[0]), pd.Timestamp(row[1]), int(row[2])


def load_duckdb_bars(
    spec: dict,
    minutes: int,
    source_min_ts: pd.Timestamp | None,
    source_max_ts: pd.Timestamp | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    db_path = Path(spec["db"])
    if not db_path.exists():
        raise SystemExit(f"Missing derived DuckDB: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        f"""
        select
          timestamp_utc,
          open,
          high,
          low,
          close,
          volume,
          source_bar_count,
          first_source_ts,
          last_source_ts,
          contains_source_gap
        from {spec["table"]}
        where source_bar_count = ?
          and contains_source_gap = false
        order by timestamp_utc
        """,
        [minutes],
    ).fetchdf()
    con.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df["first_source_ts"] = pd.to_datetime(df["first_source_ts"])
    df["last_source_ts"] = pd.to_datetime(df["last_source_ts"])
    if args.scope == "yfinance-segment":
        if source_min_ts is None or source_max_ts is None:
            raise SystemExit("No MNQ=F_YF segment exists in source 1m DB")
        df = df[
            (df["first_source_ts"] >= source_min_ts)
            & (df["last_source_ts"] <= source_max_ts)
        ].copy()
    if args.exclude_latest_bars > 0 and len(df) > args.exclude_latest_bars:
        cutoff = df["timestamp_utc"].sort_values().iloc[-args.exclude_latest_bars - 1]
        df = df[df["timestamp_utc"] <= cutoff].copy()
    return df


def compare_interval(interval: str, args: argparse.Namespace) -> dict:
    spec = TF_SPECS[interval]
    minutes = spec["minutes"]
    source_min_ts, source_max_ts, source_rows = yfinance_source_bounds(Path(args.source_1m_db))
    raw = yf.download(
        TICKER,
        period=args.period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        prepost=True,
        threads=False,
    )
    yf_df = normalize_yfinance(raw, minutes)
    db_df = load_duckdb_bars(spec, minutes, source_min_ts, source_max_ts, args)
    merged = db_df.merge(yf_df, on="timestamp_utc", how="inner")

    if merged.empty:
        return {
            "interval": interval,
            "status": "FAIL",
            "reason": "No overlap between derived DuckDB bars and Yahoo direct bars",
            "db_rows": int(len(db_df)),
            "yf_rows": int(len(yf_df)),
        }

    price_cols = ["open", "high", "low", "close"]
    mismatch_mask = pd.Series(False, index=merged.index)
    max_abs_diff: dict[str, float] = {}
    for col in price_cols:
        diff = (merged[col].astype(float) - merged[f"yf_{col}"].astype(float)).abs()
        merged[f"{col}_diff"] = diff
        max_abs_diff[col] = float(diff.max())
        mismatch_mask |= diff > args.price_tolerance
    vol_diff = (merged["volume"].astype(float) - merged["yf_volume"].astype(float)).abs()
    merged["volume_diff"] = vol_diff
    max_abs_diff["volume"] = float(vol_diff.max())
    mismatch_mask |= vol_diff > args.volume_tolerance

    mismatches = merged[mismatch_mask].copy()
    rng = np.random.default_rng(args.seed)
    sample_n = min(args.sample_size, len(merged))
    sample_idx = rng.choice(merged.index.to_numpy(), size=sample_n, replace=False) if sample_n else []
    sample = merged.loc[sample_idx].sort_values("timestamp_utc")

    mismatch_rate = float(len(mismatches) / len(merged))
    status = "PASS" if mismatch_rate <= args.max_mismatch_rate else "FAIL"
    return {
        "interval": interval,
        "status": status,
        "label_alignment": "Yahoo left-label shifted to pipeline right-label",
        "scope": args.scope,
        "source_1m_segment": {
            "source_symbol": SOURCE_SYMBOL,
            "rows": source_rows,
            "min_ts": source_min_ts.isoformat() if source_min_ts is not None else None,
            "max_ts": source_max_ts.isoformat() if source_max_ts is not None else None,
        },
        "excluded_latest_bars": args.exclude_latest_bars,
        "period": args.period,
        "db_rows_valid": int(len(db_df)),
        "yf_rows": int(len(yf_df)),
        "overlap_rows": int(len(merged)),
        "overlap_min_ts": merged["timestamp_utc"].min().isoformat(),
        "overlap_max_ts": merged["timestamp_utc"].max().isoformat(),
        "mismatch_rows": int(len(mismatches)),
        "mismatch_rate": mismatch_rate,
        "max_abs_diff": max_abs_diff,
        "first_mismatches": mismatches[
            [
                "timestamp_utc",
                "open",
                "yf_open",
                "high",
                "yf_high",
                "low",
                "yf_low",
                "close",
                "yf_close",
                "volume",
                "yf_volume",
                "open_diff",
                "high_diff",
                "low_diff",
                "close_diff",
                "volume_diff",
            ]
        ].head(20).to_dict(orient="records"),
        "random_sample": sample[
            [
                "timestamp_utc",
                "open",
                "yf_open",
                "high",
                "yf_high",
                "low",
                "yf_low",
                "close",
                "yf_close",
                "volume",
                "yf_volume",
            ]
        ].to_dict(orient="records"),
    }


def json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main() -> int:
    args = parse_args()
    results = [compare_interval(interval, args) for interval in args.intervals]
    overall_status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticker": TICKER,
        "status": overall_status,
        "scope": args.scope,
        "excluded_latest_bars": args.exclude_latest_bars,
        "price_tolerance": args.price_tolerance,
        "volume_tolerance": args.volume_tolerance,
        "max_mismatch_rate": args.max_mismatch_rate,
        "results": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
