#!/usr/bin/env python3
"""Build MNQ 1m DuckDB from Databento NDJSON OHLCV.

The Databento export can contain multiple contracts and calendar spreads. This
builder keeps outright MNQ contracts in `ohlcv_1m_contracts` and creates a
single-series `ohlcv_1m` table by selecting the highest-volume outright contract
at each timestamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/Level_0_Raw/databento/glbx-mdp3-20160501-20260518.ohlcv-1m.json"
DEFAULT_OUTPUT = ROOT / "data/Level_0_Raw/MNQ_1m.duckdb"
DEFAULT_MANIFEST = ROOT / "data/Level_0_Raw/MNQ_1m_duckdb_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    if not input_path.exists():
        raise SystemExit(f"Missing Databento input: {input_path}")
    if output_path.exists() and not args.force:
        raise SystemExit(f"Output exists; use --force to replace: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    con = duckdb.connect(str(output_path))
    con.execute("set threads to 4")
    con.execute("set preserve_insertion_order to false")

    print(f"input={input_path}")
    print(f"output={output_path}")
    print("creating ohlcv_1m_contracts...")
    con.execute(
        """
        create table ohlcv_1m_contracts as
        select
          cast(replace(replace(hd.ts_event, 'T', ' '), 'Z', '') as timestamp_ns) as timestamp_utc,
          symbol::varchar as symbol,
          hd.instrument_id::bigint as instrument_id,
          open::double as open,
          high::double as high,
          low::double as low,
          close::double as close,
          volume::bigint as volume
        from read_json_auto(?, format='newline_delimited')
        where regexp_matches(symbol, '^MNQ[HMUZ][0-9]$')
        """,
        [str(input_path)],
    )

    print("creating ohlcv_1m continuous highest-volume table...")
    con.execute(
        """
        create table ohlcv_1m as
        with ranked as (
          select
            *,
            row_number() over (
              partition by timestamp_utc
              order by volume desc, instrument_id desc, symbol asc
            ) as rn
          from ohlcv_1m_contracts
        )
        select
          timestamp_utc,
          open,
          high,
          low,
          close,
          volume,
          symbol as source_symbol,
          instrument_id
        from ranked
        where rn = 1
        order by timestamp_utc
        """
    )

    con.execute("create index idx_ohlcv_1m_ts on ohlcv_1m(timestamp_utc)")
    con.execute("create index idx_ohlcv_1m_contracts_ts_symbol on ohlcv_1m_contracts(timestamp_utc, symbol)")

    summary = con.execute(
        """
        select
          count(*)::bigint as rows,
          min(timestamp_utc)::varchar as min_ts,
          max(timestamp_utc)::varchar as max_ts,
          count(distinct source_symbol)::bigint as source_symbols
        from ohlcv_1m
        """
    ).fetchone()
    contract_summary = con.execute(
        """
        select
          count(*)::bigint as rows,
          count(distinct symbol)::bigint as symbols
        from ohlcv_1m_contracts
        """
    ).fetchone()
    top_symbols = con.execute(
        """
        select source_symbol, count(*)::bigint as rows
        from ohlcv_1m
        group by source_symbol
        order by rows desc
        limit 20
        """
    ).fetchall()
    duplicate_ts = con.execute(
        """
        select count(*)::bigint
        from (
          select timestamp_utc, count(*) as n
          from ohlcv_1m
          group by timestamp_utc
          having count(*) > 1
        )
        """
    ).fetchone()[0]
    gap_summary = con.execute(
        """
        with diffs as (
          select
            date_diff('second', lag(timestamp_utc) over (order by timestamp_utc), timestamp_utc) as step_seconds
          from ohlcv_1m
        )
        select
          count(*) filter (where step_seconds > 60)::bigint as gaps_gt_60s,
          max(step_seconds)::bigint as max_gap_seconds
        from diffs
        """
    ).fetchone()
    con.close()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "tables": {
            "ohlcv_1m": {
                "rows": int(summary[0]),
                "min_ts": summary[1],
                "max_ts": summary[2],
                "source_symbols": int(summary[3]),
                "duplicate_timestamps": int(duplicate_ts),
                "gaps_gt_60s": int(gap_summary[0] or 0),
                "max_gap_seconds": int(gap_summary[1] or 0),
                "selection_rule": "highest volume outright MNQ contract per timestamp",
            },
            "ohlcv_1m_contracts": {
                "rows": int(contract_summary[0]),
                "symbols": int(contract_summary[1]),
            },
        },
        "top_source_symbols": [
            {"symbol": symbol, "rows": int(rows)}
            for symbol, rows in top_symbols
        ],
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
