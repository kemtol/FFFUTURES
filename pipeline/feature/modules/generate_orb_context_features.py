#!/usr/bin/env python3
"""
Generate ``orb_context_features`` module parquet.

Reimplements the logic from ``build_market_context.py`` but outputs a module
parquet (grain: breakout-event level) instead of patching the datamart.

Output: ``data/Level_1_Features/modules/orb_context_features.parquet``

Features
--------
- ``orb_range_atr_ratio`` — ORB range / ATR(14) at entry
- ``day_of_week`` — 0=Mon … 4=Fri
- ``time_in_session_min`` — minutes since session open
- ``vwap_at_breakout`` — session VWAP value at breakout timestamp
- ``price_vs_vwap_pct`` — (entry_price - vwap) / vwap × 100
- ``adx_14_15m`` — ADX(14) on 15m bars at breakout
- ``ema_slope_1h`` — sign of EMA(20) slope on hourly bars (1=up, -1=down)
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_NAME = "orb_context"

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # futures/
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "Level_1_Features"
MODULES_DIR = FEATURES_DIR / "modules"
DB_1M = DATA_DIR / "Level_0_Raw" / "MGC_1m.db"
DB_15M = DATA_DIR / "Level_0_Raw" / "MGC_15m.db"
BO_PATH = FEATURES_DIR / "breakout_events.parquet"

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

SESSION_OPENS = {"tokyo": (0, 0), "london": (7, 0), "us": (13, 30)}


# ── data loaders ──────────────────────────────────────────────────────────────


def load_1m() -> pd.DataFrame:
    conn = sqlite3.connect(DB_1M)
    df = pd.read_sql(
        "SELECT timestamp_utc, high, low, close, volume "
        "FROM investing_ohlcv_1m WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.set_index("ts").sort_index()


def load_15m() -> pd.DataFrame:
    conn = sqlite3.connect(DB_15M)
    df = pd.read_sql(
        "SELECT timestamp_utc, high, low, close "
        "FROM investing_ohlcv_15m WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.set_index("ts").sort_index()


# ── indicator computation ──────────────────────────────────────────────────────


def compute_session_vwap(df1m: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP (cumulative within each session)."""
    hmin = df1m.index.hour * 60 + df1m.index.minute
    tag = pd.Series("none", index=df1m.index, dtype=str)
    tag[(hmin >= 0) & (hmin < 180)] = "tokyo"
    tag[(hmin >= 420) & (hmin < 600)] = "london"
    tag[(hmin >= 810) & (hmin < 990)] = "us"

    in_sess = tag != "none"
    group_key = df1m.index.normalize().astype(str) + "_" + tag

    typical = (df1m["high"] + df1m["low"] + df1m["close"]) / 3
    tp_vol = typical * df1m["volume"]

    tmp = pd.DataFrame({"g": group_key, "tp": tp_vol, "v": df1m["volume"]})[in_sess]
    cum_tp = tmp.groupby("g")["tp"].cumsum()
    cum_v = tmp.groupby("g")["v"].cumsum()

    vwap = pd.Series(np.nan, index=df1m.index)
    vwap[in_sess] = cum_tp / cum_v
    return vwap


def compute_adx(df15m: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX(14) on 15m bars."""
    h, l, c = df15m["high"], df15m["low"], df15m["close"]

    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)

    dm_p = pd.Series(
        np.where(
            (h - h.shift(1)) > (l.shift(1) - l),
            np.maximum((h - h.shift(1)).values, 0),
            0,
        ),
        index=df15m.index,
    )
    dm_m = pd.Series(
        np.where(
            (l.shift(1) - l) > (h - h.shift(1)),
            np.maximum((l.shift(1) - l).values, 0),
            0,
        ),
        index=df15m.index,
    )

    a = 1 / period
    atr_s = tr.ewm(alpha=a, adjust=False).mean()
    dmp_s = dm_p.ewm(alpha=a, adjust=False).mean()
    dmm_s = dm_m.ewm(alpha=a, adjust=False).mean()

    di_p = 100 * dmp_s / atr_s
    di_m = 100 * dmm_s / atr_s
    denom = (di_p + di_m).replace(0, np.nan)
    dx = 100 * (di_p - di_m).abs() / denom
    return dx.ewm(alpha=a, adjust=False).mean()


def compute_ema_slope_1h(df1m: pd.DataFrame, period: int = 20) -> pd.Series:
    """Sign of EMA(20) slope on 1h bars, forward-filled to 1m."""
    c1h = df1m["close"].resample("1h").last().dropna()
    ema = c1h.ewm(span=period, adjust=False).mean()
    slope = np.sign(ema - ema.shift(1))
    return slope.reindex(df1m.index, method="ffill")


# ── source loader ─────────────────────────────────────────────────────────────


def load_sources() -> dict[str, pd.DataFrame]:
    """Load breakout events + compute indicators needed for features."""
    bo = pd.read_parquet(BO_PATH)
    print(f"  Breakout events: {len(bo):,}")

    print("  Loading 1m / 15m data...")
    df1m = load_1m()
    df15m = load_15m()

    print("  Computing VWAP...")
    vwap_s = compute_session_vwap(df1m)

    print("  Computing ADX(14) on 15m...")
    adx_s = compute_adx(df15m)

    print("  Computing EMA slope 1h...")
    ema_s = compute_ema_slope_1h(df1m)

    return {
        "breakout_events": bo,
        "df1m": df1m,
        "df15m": df15m,
        "vwap": vwap_s,
        "adx": adx_s,
        "ema": ema_s,
    }


# ── feature builder ───────────────────────────────────────────────────────────


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute ORB context features at breakout-event grain.

    Returns DataFrame with EVENT_KEY columns + 7 context feature columns (float32).
    """
    bo = sources["breakout_events"]
    df1m = sources["df1m"]
    vwap_s = sources["vwap"]
    adx_s = sources["adx"]
    ema_s = sources["ema"]

    # Convert to numpy for fast searchsorted lookup
    ep_1m = df1m.index.view(np.int64) // 1_000_000
    ep_15m = sources["df15m"].index.view(np.int64) // 1_000_000
    vwap_a = vwap_s.values
    adx_a = adx_s.values
    ema_a = ema_s.values

    records = []
    for _, row in bo.iterrows():
        bep = int(row["breakout_ts"].timestamp() * 1000)

        i1 = max(0, np.searchsorted(ep_1m, bep, side="right") - 1)
        i15 = max(0, np.searchsorted(ep_15m, bep, side="right") - 1)

        vwap_val = vwap_a[i1]
        entry = row["entry_price"]
        pvwap = (
            (entry - vwap_val) / vwap_val * 100
            if not np.isnan(vwap_val)
            else np.nan
        )

        sess = row["session"]
        oh, om = SESSION_OPENS[sess]
        sess_open = pd.Timestamp(str(row["date"]), tz="UTC").replace(
            hour=oh, minute=om
        )
        t_in_sess = (row["breakout_ts"] - sess_open).total_seconds() / 60

        records.append(
            {
                "date": row["date"],
                "session": sess,
                "orb_tf": row["orb_tf"],
                "breakout_ts": row["breakout_ts"],
                "orb_range_atr_ratio": (
                    row["orb_range"] / row["atr14_at_entry"]
                    if row["atr14_at_entry"] > 0
                    else np.nan
                ),
                "day_of_week": row["breakout_ts"].dayofweek,
                "time_in_session_min": t_in_sess,
                "vwap_at_breakout": vwap_val,
                "price_vs_vwap_pct": pvwap,
                "adx_14_15m": adx_a[i15],
                "ema_slope_1h": ema_a[i1],
            }
        )

    result = pd.DataFrame(records)

    # ── Ensure float32 dtypes ─────────────────────────────────────
    feat_cols = [c for c in result.columns if c not in EVENT_KEY]
    for col in feat_cols:
        result[col] = result[col].astype("float32")

    return result


# ── conflict check ────────────────────────────────────────────────────────────


def check_column_conflict(df: pd.DataFrame) -> list[str]:
    feat_cols = set(c for c in df.columns if c not in EVENT_KEY)
    conflicts: list[str] = []
    for fpath in sorted(MODULES_DIR.glob("*_features.parquet")):
        existing_cols = set(pd.read_parquet(fpath).columns)
        existing_feats = existing_cols - set(EVENT_KEY)
        overlap = feat_cols & existing_feats
        if overlap:
            conflicts.append(f"  ⚠️  {fpath.name}: {sorted(overlap)}")
    return conflicts


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Generate {MODULE_NAME} feature module"
    )
    parser.add_argument(
        "--modules-dir",
        type=Path,
        default=MODULES_DIR,
        help=f"Output directory (default: {MODULES_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write even if column name conflicts detected",
    )
    args = parser.parse_args()

    print(f"[{MODULE_NAME}] Loading sources...")
    sources = load_sources()

    print(f"[{MODULE_NAME}] Building features...")
    df = build_features(sources)

    # ── Validate ──────────────────────────────────────────────────
    n_feats = len([c for c in df.columns if c not in EVENT_KEY])
    print(f"[{MODULE_NAME}] Rows: {len(df):,}")
    print(f"[{MODULE_NAME}] Features: {n_feats}")
    print(f"[{MODULE_NAME}] Grain: ({', '.join(EVENT_KEY)})")
    print(f"[{MODULE_NAME}] Columns: {list(df.columns)}")

    for col in df.columns:
        if col in EVENT_KEY:
            continue
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 50:
            print(f"  ⚠️  {col}: {nan_pct:.1f}% NaN — may degrade model")

    conflicts = check_column_conflict(df)
    if conflicts:
        print(f"[{MODULE_NAME}] ⚠️  Column name conflicts detected:")
        for c in conflicts:
            print(c)
        if not args.force:
            print(
                f"[{MODULE_NAME}] Aborting. Use --force to write anyway, "
                f"or rename columns."
            )
            return

    if args.dry_run:
        print(f"[{MODULE_NAME}] Dry run — not writing")
        return

    # ── Write ─────────────────────────────────────────────────────
    args.modules_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.modules_dir / f"{MODULE_NAME}_features.parquet"
    df.to_parquet(out_path, index=False)
    file_size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{MODULE_NAME}] Written to {out_path} ({file_size_mb:.1f} MB)")
    print(f"[{MODULE_NAME}] ✅ Ready — just re-run sweep script (auto-detected)")


if __name__ == "__main__":
    main()
