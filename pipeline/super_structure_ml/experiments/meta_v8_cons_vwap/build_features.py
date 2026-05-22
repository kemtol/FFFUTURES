#!/usr/bin/env python3
"""Build research-only Meta-v8 CONS VWAP candidate features.

This script enriches the sterile Meta-v7 CONS datamart with causal VWAP
features. It writes a new parquet only; it does not touch live models or live
router code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def load_base_datamart(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    return df.sort_values("entry_ts").reset_index(drop=True)


def load_vwap_context(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Missing L1 VWAP cache: {path}\n"
            "Run: python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/build_l1_vwap.py"
        )
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def add_vwap_features(base: pd.DataFrame, vwap: pd.DataFrame) -> pd.DataFrame:
    out = pd.merge_asof(
        base.sort_values("entry_ts"),
        vwap.sort_values("timestamp_utc"),
        left_on="entry_ts",
        right_on="timestamp_utc",
        direction="backward",
    )
    side_mult = np.where(out["side"].astype(str).str.lower().eq("long"), 1.0, -1.0)
    atr = out["entry_atr"].astype(float).replace(0, np.nan)
    dist = out["entry_price"].astype(float) - out["ct_vwap"].astype(float)
    out["dist_to_ct_vwap_atr"] = (dist / (atr + 1e-9)).replace([np.inf, -np.inf], np.nan)
    out["vwap_side_aligned"] = ((dist * side_mult) > 0).astype(int)
    out["ct_vwap_slope_20_atr"] = (
        out["ct_vwap_slope_20"].astype(float) / (atr + 1e-9)
    ).replace([np.inf, -np.inf], np.nan)
    out["vwap_deviation_z_50"] = out["vwap_deviation_z_50"].clip(-5, 5)
    return out.drop(columns=["timestamp_utc"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    base_path = ROOT / cfg["baseline"]["datamart"]
    l1_path = ROOT / cfg["level1"]["vwap_context"]
    out_path = ROOT / cfg["candidate"]["datamart"]
    if out_path.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"Output exists: {out_path} (use --force)")

    base = load_base_datamart(base_path)
    vwap = load_vwap_context(l1_path)
    enriched = add_vwap_features(base, vwap)

    candidate_features = cfg["candidate"]["features"]
    missing = [c for c in candidate_features if c not in enriched.columns]
    if missing:
        raise SystemExit(f"Missing candidate features: {missing}")

    print(f"Base rows: {len(base):,}")
    print(f"Output rows: {len(enriched):,}")
    print("Candidate feature NaN rates:")
    print(enriched[candidate_features].isna().mean().sort_values(ascending=False).to_string())
    if args.dry_run:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
