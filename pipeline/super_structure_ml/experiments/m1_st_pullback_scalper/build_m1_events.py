#!/usr/bin/env python3
"""Build M1 SuperTrend pullback scalper events.

Research-only L2 datamart builder. It derives events from the L1 M1 context
parquet, which contains raw OHLCV + causal indicators. It does not use 5m CONS
signals and does not touch live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("/home/kemal/futures")
from pipeline.live.super_structure import _atr, adx, cci, dema, supertrend  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = Path(__file__).resolve().with_name("config.json")
L1_REQUIRED_COLUMNS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "st",
    "st_direction",
    "prev_st_direction",
    "atr",
    "entry_adx",
    "entry_cci",
    "rsi_7",
    "dema_50",
    "dema_100",
    "dema_200",
    "ct_vwap",
    "ct_vwap_slope_20",
    "vwap_deviation_z_50",
    "st_slope_5_atr",
    "close_slope_3_atr",
    "close_slope_5_atr",
    "prev_gap_seconds",
    "data_quality_ok",
]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_cluster(ts: pd.Timestamp) -> int:
    hour = int(ts.hour)
    if hour <= 6:
        return 0
    if hour <= 12:
        return 1
    return 2


def topstep_trade_day(ts: pd.Series) -> pd.Series:
    ts_ct = ts.dt.tz_convert("America/Chicago")
    return (ts_ct - pd.Timedelta(hours=15, minutes=10)).dt.date


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    avg_up = pd.Series(up).rolling(period).mean()
    avg_down = pd.Series(down).rolling(period).mean()
    rs = avg_up / (avg_down + 1e-9)
    return (100 - (100 / (1 + rs))).values


def load_1m_bars(cfg: dict, start: str | None, end: str | None) -> pd.DataFrame:
    source = cfg["source"]
    db_path = ROOT / source["db"]
    start = start or source["start_date"]

    where = [
        "symbol = ?",
        "timeframe = ?",
        "timestamp_utc >= ?",
    ]
    params: list[object] = [source["symbol"], source["timeframe"], start]
    if end:
        where.append("timestamp_utc < ?")
        params.append(end)

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30) as conn:
        df = pd.read_sql(
            f"""
            SELECT timestamp_utc, open, high, low, close, volume
            FROM {source["table"]}
            WHERE {" AND ".join(where)}
            ORDER BY epoch_ms
            """,
            conn,
            params=params,
        )

    if df.empty:
        raise RuntimeError(f"No M1 bars found for start={start} end={end}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.reset_index(drop=True)


def load_l1_context(cfg: dict, start: str | None, end: str | None) -> pd.DataFrame:
    path = ROOT / cfg["outputs"]["l1_context"]
    if not path.exists():
        raise SystemExit(
            f"Missing L1 context: {path}\n"
            "Run: python3 pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/build_l1_context.py --force"
        )
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    missing = [c for c in L1_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"L1 context missing required columns: {missing}")
    if start:
        df = df[df["timestamp_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp_utc"] < pd.Timestamp(end, tz="UTC")]
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ind = cfg["indicators"]
    out = df.copy()
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    st, direction = supertrend(
        h,
        l,
        c,
        factor=float(ind["supertrend_factor"]),
        atr_period=int(ind["atr_period"]),
    )
    out["st"] = st
    out["st_direction"] = direction
    out["prev_st_direction"] = out["st_direction"].shift(1)
    out["atr"] = _atr(h, l, c, int(ind["atr_period"]))
    out["entry_adx"] = adx(h, l, c, int(ind["adx_length"]))
    out["entry_cci"] = cci(
        h,
        l,
        c,
        int(ind["cci_length"]),
        source=str(ind["cci_source"]),
    )
    out["rsi_7"] = rsi(c, int(ind["rsi_length"]))
    out["dema_50"] = dema(c, int(ind["dema_fast"]))
    out["dema_100"] = dema(c, int(ind["dema_mid"]))
    out["dema_200"] = dema(c, int(ind["dema_slow"]))

    out["ct_trade_day"] = topstep_trade_day(out["timestamp_utc"])
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    volume = out["volume"].fillna(0).astype(float).clip(lower=0)
    weight = volume.where(volume > 0, 1.0)
    out["_pv"] = typical * weight
    out["_weight"] = weight
    grouped = out.groupby("ct_trade_day", sort=False)
    out["ct_vwap"] = grouped["_pv"].cumsum() / grouped["_weight"].cumsum()
    out["ct_vwap_slope_20"] = grouped["ct_vwap"].diff(20)
    out["_vwap_dist"] = out["close"] - out["ct_vwap"]
    out["vwap_deviation_z_50"] = grouped["_vwap_dist"].transform(
        lambda s: (s - s.rolling(50, min_periods=20).mean())
        / (s.rolling(50, min_periods=20).std() + 1e-9)
    )
    atr_safe = out["atr"].astype(float) + 1e-9
    out["st_slope_5_atr"] = out["st"].diff(5) / atr_safe
    out["close_slope_3_atr"] = out["close"].diff(3) / atr_safe
    out["close_slope_5_atr"] = out["close"].diff(5) / atr_safe
    return out


def candidate_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    rules = cfg["candidate_rules"]
    atr = df["atr"].astype(float)
    band = np.maximum(float(rules["min_pullback_band_pts"]), atr * float(rules["pullback_band_atr"]))
    df["pullback_band"] = band

    valid = (
        df["st"].notna()
        & df["atr"].notna()
        & df["dema_200"].notna()
        & df["prev_st_direction"].notna()
        & (df["st_direction"] == df["prev_st_direction"])
        & (df["entry_adx"] >= float(rules["min_adx"]))
        & df.get("data_quality_ok", True)
    )

    long_ok = (
        valid
        & (df["st_direction"] == -1)
        & (df["close"] > df["st"])
        & (df["low"] <= df["st"] + df["pullback_band"])
        & (df["close"] > df["open"])
        & (df["rsi_7"] < float(rules["skip_long_rsi_gte"]))
        & (df["entry_cci"] < float(rules["skip_long_cci_gte"]))
    )
    short_ok = (
        valid
        & (df["st_direction"] == 1)
        & (df["close"] < df["st"])
        & (df["high"] >= df["st"] - df["pullback_band"])
        & (df["close"] < df["open"])
        & (df["rsi_7"] > float(rules["skip_short_rsi_lte"]))
        & (df["entry_cci"] > float(rules["skip_short_cci_lte"]))
    )

    if bool(rules["use_dema100_trend_filter"]):
        long_ok &= df["close"] > df["dema_100"]
        short_ok &= df["close"] < df["dema_100"]

    return long_ok | short_ok


def add_data_quality_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    quality = cfg["data_quality"]
    out = df.copy()
    expected_step = float(quality["expected_step_seconds"])
    short_gap_limit = float(quality["short_gap_fail_seconds"])
    quarantine_bars = int(quality["quarantine_bars_after_unexpected_gap"])

    out["prev_gap_seconds"] = out["timestamp_utc"].diff().dt.total_seconds().fillna(expected_step)
    out["unexpected_short_gap"] = (
        (out["prev_gap_seconds"] > expected_step)
        & (out["prev_gap_seconds"] < short_gap_limit)
    )
    out["data_quality_ok"] = ~(
        out["unexpected_short_gap"]
        .rolling(quarantine_bars + 1, min_periods=1)
        .max()
        .astype(bool)
    )
    return out


def conditional_outcome(
    *,
    side: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    future: pd.DataFrame,
    cfg: dict,
) -> tuple[int, float, str, pd.Timestamp, float, int]:
    costs = cfg["costs"]
    exits = cfg["exit_rules"]
    point_value = float(costs["point_value"])
    commission = float(costs["commission_round_turn_usd"])
    slippage_pts = float(costs["slippage_pts"])

    if future.empty:
        return 0, -commission, "NO_FUTURE", pd.NaT, entry_price, 0

    for hold_bars, (_, bar) in enumerate(future.iterrows(), start=1):
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        ts = pd.Timestamp(bar["timestamp_utc"])

        if side == "Long":
            if low <= sl_price:
                exit_price = sl_price - slippage_pts
                pnl_pts = exit_price - entry_price
                return 0, pnl_pts * point_value - commission, "SL", ts, exit_price, hold_bars
            if high >= tp_price:
                exit_price = tp_price - slippage_pts
                pnl_pts = exit_price - entry_price
                return 1, pnl_pts * point_value - commission, "TP_R", ts, exit_price, hold_bars
            in_profit = close > entry_price
            exhausted = (
                float(bar["rsi_7"]) >= float(exits["take_profit_long_rsi_gte"])
                or float(bar["entry_cci"]) >= float(exits["take_profit_long_cci_gte"])
            )
            if in_profit and exhausted:
                exit_price = close - slippage_pts
                pnl_pts = exit_price - entry_price
                return int(pnl_pts > 0), pnl_pts * point_value - commission, "TP_MOMENTUM", ts, exit_price, hold_bars
            if int(bar["st_direction"]) == 1:
                exit_price = close - slippage_pts
                pnl_pts = exit_price - entry_price
                return int(pnl_pts > 0), pnl_pts * point_value - commission, "ST_FLIP", ts, exit_price, hold_bars
        else:
            if high >= sl_price:
                exit_price = sl_price + slippage_pts
                pnl_pts = entry_price - exit_price
                return 0, pnl_pts * point_value - commission, "SL", ts, exit_price, hold_bars
            if low <= tp_price:
                exit_price = tp_price + slippage_pts
                pnl_pts = entry_price - exit_price
                return 1, pnl_pts * point_value - commission, "TP_R", ts, exit_price, hold_bars
            in_profit = close < entry_price
            exhausted = (
                float(bar["rsi_7"]) <= float(exits["take_profit_short_rsi_lte"])
                or float(bar["entry_cci"]) <= float(exits["take_profit_short_cci_lte"])
            )
            if in_profit and exhausted:
                exit_price = close + slippage_pts
                pnl_pts = entry_price - exit_price
                return int(pnl_pts > 0), pnl_pts * point_value - commission, "TP_MOMENTUM", ts, exit_price, hold_bars
            if int(bar["st_direction"]) == -1:
                exit_price = close + slippage_pts
                pnl_pts = entry_price - exit_price
                return int(pnl_pts > 0), pnl_pts * point_value - commission, "ST_FLIP", ts, exit_price, hold_bars

    last = future.iloc[-1]
    close = float(last["close"])
    ts = pd.Timestamp(last["timestamp_utc"])
    exit_price = close - slippage_pts if side == "Long" else close + slippage_pts
    pnl_pts = exit_price - entry_price if side == "Long" else entry_price - exit_price
    return int(pnl_pts > 0), pnl_pts * point_value - commission, "TIMEOUT", ts, exit_price, len(future)


def build_events(df: pd.DataFrame, cfg: dict, limit_events: int | None = None) -> pd.DataFrame:
    rules = cfg["candidate_rules"]
    exits = cfg["exit_rules"]
    costs = cfg["costs"]
    execution = cfg["execution"]
    max_hold = int(exits["max_hold_bars"])
    st_buffer = float(exits["st_buffer_pts"])
    rr_target = float(exits["rr_target"])
    slippage_pts = float(costs["slippage_pts"])
    max_entry_gap = pd.Timedelta(minutes=float(execution["max_entry_gap_minutes"]))
    reject_outcome_gap = float(cfg["data_quality"]["reject_outcome_window_gap_gt_seconds"])

    df = add_data_quality_flags(df, cfg)
    signals = df[candidate_mask(df, cfg)].copy()
    print(f"Raw candidates before risk/outcome simulation: {len(signals):,}")
    if limit_events:
        signals = signals.head(limit_events)
        print(f"Limiting outcome simulation to first {len(signals):,} events")

    events = []
    for idx, row in signals.iterrows():
        entry_idx = idx + 1
        if entry_idx >= len(df):
            continue
        entry_row = df.iloc[entry_idx]
        entry_gap = pd.Timestamp(entry_row["timestamp_utc"]) - pd.Timestamp(row["timestamp_utc"])
        if entry_gap <= pd.Timedelta(0) or entry_gap > max_entry_gap:
            continue

        side = "Long" if int(row["st_direction"]) == -1 else "Short"
        entry_open = float(entry_row["open"])
        entry = entry_open + slippage_pts if side == "Long" else entry_open - slippage_pts
        if side == "Long":
            sl = float(row["st"]) - st_buffer
            risk = entry - sl
            tp = entry + risk * rr_target
            touch_distance_atr = (float(row["low"]) - float(row["st"])) / (float(row["atr"]) + 1e-9)
        else:
            sl = float(row["st"]) + st_buffer
            risk = sl - entry
            tp = entry - risk * rr_target
            touch_distance_atr = (float(row["st"]) - float(row["high"])) / (float(row["atr"]) + 1e-9)

        if risk < float(rules["min_risk_pts"]) or risk > float(rules["max_risk_pts"]):
            continue

        future = df.iloc[entry_idx:entry_idx + max_hold]
        if (future["prev_gap_seconds"] > reject_outcome_gap).any():
            continue
        label, pnl_usd, exit_reason, exit_ts, exit_price, hold_bars = conditional_outcome(
            side=side,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            future=future,
            cfg=cfg,
        )

        atr_safe = float(row["atr"]) + 1e-9
        body = abs(float(row["close"]) - float(row["open"]))
        bar_range = float(row["high"]) - float(row["low"])
        close_pos = (float(row["close"]) - float(row["low"])) / (bar_range + 1e-9)

        events.append(
            {
                "event_id": f"M1ST_{row['timestamp_utc'].strftime('%Y%m%d%H%M')}_{side}",
                "signal_ts": row["timestamp_utc"],
                "entry_ts": entry_row["timestamp_utc"],
                "side": side,
                "signal_open": float(row["open"]),
                "signal_high": float(row["high"]),
                "signal_low": float(row["low"]),
                "signal_close": float(row["close"]),
                "signal_volume": float(row["volume"]) if pd.notna(row["volume"]) else 0.0,
                "entry_open": float(entry_row["open"]),
                "entry_high": float(entry_row["high"]),
                "entry_low": float(entry_row["low"]),
                "entry_close": float(entry_row["close"]),
                "entry_volume": float(entry_row["volume"]) if pd.notna(entry_row["volume"]) else 0.0,
                "entry_gap_seconds": float(entry_gap.total_seconds()),
                "signal_prev_gap_seconds": float(row["prev_gap_seconds"]),
                "signal_data_quality_ok": bool(row["data_quality_ok"]),
                "entry_price": entry,
                "sl_price": sl,
                "tp_price": tp,
                "exit_ts": exit_ts,
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "hold_bars": int(hold_bars),
                "risk_pts": float(risk),
                "label": int(label),
                "pnl_usd": float(pnl_usd),
                "signal_adx": float(row["entry_adx"]),
                "signal_cci": float(row["entry_cci"]),
                "cci_abs": float(abs(row["entry_cci"])),
                "signal_rsi_7": float(row["rsi_7"]),
                "signal_atr": float(row["atr"]),
                "signal_st": float(row["st"]),
                "signal_st_direction": int(row["st_direction"]),
                "signal_dema_50": float(row["dema_50"]),
                "signal_dema_100": float(row["dema_100"]),
                "signal_dema_200": float(row["dema_200"]),
                "signal_ct_vwap": float(row["ct_vwap"]),
                "signal_ct_vwap_slope_20": float(row["ct_vwap_slope_20"])
                if pd.notna(row["ct_vwap_slope_20"])
                else 0.0,
                "st_gap_atr": float(abs(row["close"] - row["st"]) / atr_safe),
                "st_slope_5_atr": float(row["st_slope_5_atr"]) if pd.notna(row["st_slope_5_atr"]) else 0.0,
                "touch_distance_atr": float(touch_distance_atr),
                "pullback_band_atr": float(row["pullback_band"] / atr_safe),
                "dist_d50_atr": float((row["close"] - row["dema_50"]) / atr_safe),
                "dist_d100_atr": float((row["close"] - row["dema_100"]) / atr_safe),
                "dist_d200_atr": float((row["close"] - row["dema_200"]) / atr_safe),
                "dema_stack": int(
                    3
                    if row["close"] > row["dema_50"] > row["dema_100"] > row["dema_200"]
                    else -3
                    if row["close"] < row["dema_50"] < row["dema_100"] < row["dema_200"]
                    else 0
                ),
                "close_slope_3_atr": float(row["close_slope_3_atr"]) if pd.notna(row["close_slope_3_atr"]) else 0.0,
                "close_slope_5_atr": float(row["close_slope_5_atr"]) if pd.notna(row["close_slope_5_atr"]) else 0.0,
                "wick_ratio": float((bar_range - body) / (bar_range + 1e-9)),
                "candle_body_atr": float(body / atr_safe),
                "bar_range_atr": float(bar_range / atr_safe),
                "directional_close_pos": float(close_pos if side == "Long" else 1.0 - close_pos),
                "dist_to_ct_vwap_atr": float((row["close"] - row["ct_vwap"]) / atr_safe),
                "ct_vwap_slope_20_atr": float(row["ct_vwap_slope_20"] / atr_safe)
                if pd.notna(row["ct_vwap_slope_20"])
                else 0.0,
                "vwap_deviation_z_50": float(np.clip(row["vwap_deviation_z_50"], -5, 5))
                if pd.notna(row["vwap_deviation_z_50"])
                else 0.0,
                "hour_utc": int(row["timestamp_utc"].hour),
                "dow": int(row["timestamp_utc"].dayofweek),
                "session_cluster": session_cluster(row["timestamp_utc"]),
                "point_value": float(costs["point_value"]),
                "commission_usd": float(costs["commission_round_turn_usd"]),
                "slippage_pts": float(costs["slippage_pts"]),
            }
        )

    return pd.DataFrame(events)


def validate_events(events: pd.DataFrame, context: pd.DataFrame) -> None:
    if events.empty:
        raise ValueError("L2 events datamart is empty")
    if events["event_id"].duplicated().any():
        raise ValueError(f"Duplicate event_id rows: {int(events['event_id'].duplicated().sum())}")
    if not events["entry_ts"].is_monotonic_increasing:
        raise ValueError("L2 entry_ts is not sorted ascending")
    if (events["entry_ts"] <= events["signal_ts"]).any():
        raise ValueError("L2 contains entry_ts <= signal_ts")
    if "signal_data_quality_ok" in events.columns and (~events["signal_data_quality_ok"]).any():
        raise ValueError("L2 contains events from quarantined data-quality windows")
    if (events["risk_pts"] <= 0).any():
        raise ValueError("L2 contains non-positive risk_pts")
    if (events["hold_bars"] < 0).any():
        raise ValueError("L2 contains negative hold_bars")

    ctx = context.set_index("timestamp_utc")
    missing_signal_ts = ~events["signal_ts"].isin(ctx.index)
    if missing_signal_ts.any():
        raise ValueError(f"L2 signal_ts missing from L1 context: {int(missing_signal_ts.sum())}")
    missing_entry_ts = ~events["entry_ts"].isin(ctx.index)
    if missing_entry_ts.any():
        raise ValueError(f"L2 entry_ts missing from L1 context: {int(missing_entry_ts.sum())}")

    signal_joined = events.join(
        ctx[["open", "high", "low", "close", "volume"]],
        on="signal_ts",
        rsuffix="_l1",
    )
    signal_checks = {
        "signal_open": "open",
        "signal_high": "high",
        "signal_low": "low",
        "signal_close": "close",
        "signal_volume": "volume",
    }
    for event_col, l1_col in signal_checks.items():
        diff = (signal_joined[event_col].astype(float) - signal_joined[l1_col].astype(float).fillna(0.0)).abs()
        if (diff > 1e-9).any():
            raise ValueError(f"L2 {event_col} mismatch vs L1 signal {l1_col}: {int((diff > 1e-9).sum())}")

    entry_joined = events.join(
        ctx[["open", "high", "low", "close", "volume"]],
        on="entry_ts",
        rsuffix="_l1",
    )
    entry_checks = {
        "entry_open": "open",
        "entry_high": "high",
        "entry_low": "low",
        "entry_close": "close",
        "entry_volume": "volume",
    }
    for event_col, l1_col in entry_checks.items():
        diff = (entry_joined[event_col].astype(float) - entry_joined[l1_col].astype(float).fillna(0.0)).abs()
        if (diff > 1e-9).any():
            raise ValueError(f"L2 {event_col} mismatch vs L1 entry {l1_col}: {int((diff > 1e-9).sum())}")


def write_events_manifest(path: Path, manifest_path: Path, events: pd.DataFrame, cfg: dict) -> None:
    manifest = {
        "artifact": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "rows": int(len(events)),
        "columns": list(events.columns),
        "signal_ts_min": events["signal_ts"].min().isoformat(),
        "signal_ts_max": events["signal_ts"].max().isoformat(),
        "entry_ts_min": events["entry_ts"].min().isoformat(),
        "entry_ts_max": events["entry_ts"].max().isoformat(),
        "source_l1_context": cfg["outputs"]["l1_context"],
        "execution": cfg["execution"],
        "data_quality": cfg["data_quality"],
        "candidate_rules": cfg["candidate_rules"],
        "exit_rules": cfg["exit_rules"],
        "costs": cfg["costs"],
        "null_rates": {
            c: float(v)
            for c, v in events.isna().mean().sort_values(ascending=False).items()
            if v > 0
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def print_summary(events: pd.DataFrame) -> None:
    print(f"Events: {len(events):,}")
    if events.empty:
        return
    print(f"Signal range: {events['signal_ts'].min()} -> {events['signal_ts'].max()}")
    print(f"Entry range: {events['entry_ts'].min()} -> {events['entry_ts'].max()}")
    print("By side:")
    print(events.groupby("side").agg(events=("event_id", "size"), pnl=("pnl_usd", "sum"), avg=("pnl_usd", "mean")).to_string())
    print("By exit reason:")
    print(events.groupby("exit_reason").agg(events=("event_id", "size"), pnl=("pnl_usd", "sum"), avg=("pnl_usd", "mean")).to_string())
    print("Overall:")
    print(
        pd.Series(
            {
                "pnl_usd": events["pnl_usd"].sum(),
                "avg_trade": events["pnl_usd"].mean(),
                "win_rate": (events["pnl_usd"] > 0).mean(),
                "avg_hold_bars": events["hold_bars"].mean(),
            }
        ).to_string()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--limit-events", type=int)
    parser.add_argument("--rebuild-context", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    out_path = ROOT / cfg["outputs"]["events"]
    manifest_path = ROOT / cfg["outputs"]["events_manifest"]
    if out_path.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"Output exists: {out_path} (use --force)")

    if args.rebuild_context:
        bars = load_1m_bars(cfg, args.start_date, args.end_date)
        enriched = add_indicators(bars, cfg)
    else:
        enriched = load_l1_context(cfg, args.start_date, args.end_date)
    events = build_events(enriched, cfg, limit_events=args.limit_events)
    validate_events(events, enriched)
    print_summary(events)

    if args.dry_run:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_path, index=False)
    write_events_manifest(out_path, manifest_path, events, cfg)
    print(f"Wrote {out_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
