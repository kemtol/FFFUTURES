#!/usr/bin/env python3
"""Build trade-event datamart for Super Structure (ST + DEMA + ADX + CCI) strategy.

This is the bridge between a simple backtest and setup/probability research:
one output row per completed trade, with entry-time features and realized
outcomes such as PnL, R multiple, MFE, and MAE.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.super_structure import (
    ADX_LENGTH,
    ADX_THRESHOLD,
    ATR_PERIOD,
    CCI_LENGTH,
    CCI_LONG_MIN,
    CCI_SHORT_MAX,
    CCI_SOURCE,
    DEMA_LENGTH,
    ST_FACTOR,
    _atr,
    adx,
    cci,
    dema,
    supertrend,
)
from pipeline.live.pullback_detector import (
    detect_pullback_event,
    MAX_HOLD_BARS as AGGR_MAX_HOLD_BARS,
)

RAW_DB = ROOT / "data/Level_0_Raw/MGC_1m.db"
OUT_PARQUET = ROOT / "data/Level_2_Datamart/super_structure_trade_events.parquet"
OUT_CANDLES_JSON = ROOT / "ui/data/candles_super_structure_5m.json"
OUT_TRADES_JSON = ROOT / "ui/data/trade_events_super_structure.json"
STRATEGY_KEY = "super_structure"

SYMBOL = "MICRO_GOLD"
TIMEFRAME = "1m"
POINT_VALUE_USD = 10.0
ROUND_TURN_COMMISSION_USD = 1.74
WARMUP_DAYS = 120


@dataclass
class OpenTrade:
    side: str
    entry_ts: pd.Timestamp
    entry_i: int
    entry_price: float
    initial_sl: float
    sl_price: float
    entry_adx: float
    entry_cci: float
    entry_dema: float
    entry_supertrend: float
    entry_direction: int
    entry_atr: float
    entry_bar_high: float
    entry_bar_low: float
    entry_bar_close: float
    mfe_points: float = 0.0
    mae_points: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-01-01", help="event start date UTC")
    parser.add_argument("--end", default="2026-05-01", help="event end date UTC, exclusive")
    parser.add_argument("--raw-db", type=Path, default=RAW_DB)
    parser.add_argument("--table", default="investing_ohlcv_1m", help="Source table name")
    parser.add_argument("--out", type=Path, default=OUT_PARQUET)
    parser.add_argument("--export-ui", action="store_true", help="write ui/data JSON files")
    return parser.parse_args()


def load_ohlcv_1m(db_path: Path, start: str, end: str, table: str = "investing_ohlcv_1m") -> pd.DataFrame:
    warmup_start = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    end_ts = pd.Timestamp(end, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql(
            f"""
            SELECT timestamp_utc, open, high, low, close, volume
            FROM {table}
            WHERE symbol = ? AND timeframe = ?
              AND timestamp_utc >= ? AND timestamp_utc < ?
            ORDER BY epoch_ms
            """,
            conn,
            params=[SYMBOL, TIMEFRAME, warmup_start, end_ts],
        )
    if df.empty:
        raise RuntimeError(f"No OHLCV rows found in {db_path} for {warmup_start} -> {end_ts}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df_1m.set_index("timestamp_utc").sort_index()
    out = df.resample(rule, label="right", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    return out


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlcv(df_1m, "5min")


def session_name(ts: pd.Timestamp) -> str:
    hour = ts.hour + ts.minute / 60.0
    if 0 <= hour < 3:
        return "Tokyo"
    if 7 <= hour < 10:
        return "London"
    if 13.5 <= hour < 16.5:
        return "US"
    return "Other"


def update_excursions(trade: OpenTrade, high: float, low: float) -> None:
    if trade.side == "Long":
        fav = high - trade.entry_price
        adv = low - trade.entry_price
    else:
        fav = trade.entry_price - low
        adv = trade.entry_price - high
    trade.mfe_points = max(trade.mfe_points, fav)
    trade.mae_points = min(trade.mae_points, adv)


def close_trade(
    trade: OpenTrade,
    exit_ts: pd.Timestamp,
    exit_i: int,
    exit_price: float,
    exit_reason: str,
    bars_held: int,
    timeframe_min: int,
) -> dict:
    gross_points = exit_price - trade.entry_price if trade.side == "Long" else trade.entry_price - exit_price
    gross_usd = gross_points * POINT_VALUE_USD
    pnl_usd = gross_usd - ROUND_TURN_COMMISSION_USD
    risk_points = abs(trade.entry_price - trade.initial_sl)
    risk_usd = risk_points * POINT_VALUE_USD
    r_multiple = gross_usd / risk_usd if risk_usd > 0 else np.nan

    dema_distance = (
        trade.entry_price - trade.entry_dema
        if trade.side == "Long"
        else trade.entry_dema - trade.entry_price
    )
    st_distance = abs(trade.entry_price - trade.entry_supertrend)

    return {
        "entry_ts": trade.entry_ts,
        "exit_ts": exit_ts,
        "side": trade.side,
        "entry_price": trade.entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "duration_min": bars_held * timeframe_min,
        "gross_points": gross_points,
        "gross_usd": gross_usd,
        "commission_usd": ROUND_TURN_COMMISSION_USD,
        "pnl_usd": pnl_usd,
        "risk_points": risk_points,
        "risk_usd": risk_usd,
        "r_multiple": r_multiple,
        "mfe_points": trade.mfe_points,
        "mae_points": trade.mae_points,
        "mfe_usd": trade.mfe_points * POINT_VALUE_USD,
        "mae_usd": trade.mae_points * POINT_VALUE_USD,
        "entry_adx": trade.entry_adx,
        "entry_cci": trade.entry_cci,
        "entry_dema": trade.entry_dema,
        "entry_supertrend": trade.entry_supertrend,
        "entry_st_direction": trade.entry_direction,
        "entry_atr": trade.entry_atr,
        "dema_distance": dema_distance,
        "dema_distance_atr": dema_distance / trade.entry_atr if trade.entry_atr > 0 else np.nan,
        "st_distance": st_distance,
        "st_distance_atr": st_distance / trade.entry_atr if trade.entry_atr > 0 else np.nan,
        "entry_bar_high": trade.entry_bar_high,
        "entry_bar_low": trade.entry_bar_low,
        "entry_bar_close": trade.entry_bar_close,
        "hour_utc": trade.entry_ts.hour,
        "minute_utc": trade.entry_ts.minute,
        "session": session_name(trade.entry_ts),
        "is_win": pnl_usd > 0,
        "hit_1r": trade.mfe_points >= risk_points if risk_points > 0 else False,
        "hit_2r": trade.mfe_points >= 2 * risk_points if risk_points > 0 else False,
        "entry_i": trade.entry_i,
        "exit_i": exit_i,
    }


def build_aggr_pullback_events(df_bars: pd.DataFrame, event_start: str,
                                timeframe_min: int) -> pd.DataFrame:
    """Generate AGGR (v1.12 pullback) theoretical trades for UI overlay.

    Mirrors `pipeline/live/pullback_detector.py` + v1.12 datamart builder
    so live AGGR trades show up in the UI/parity stream. Single-queue rule
    is NOT applied here — UI shows ALL theoretical AGGR opportunities for
    research visibility.
    """
    h = df_bars["high"].to_numpy(dtype=float)
    l_arr = df_bars["low"].to_numpy(dtype=float)
    c = df_bars["close"].to_numpy(dtype=float)
    o = df_bars["open"].to_numpy(dtype=float)

    st, direction = supertrend(h, l_arr, c, ST_FACTOR, ATR_PERIOD)
    d100 = dema(c, 100)
    d200 = dema(c, DEMA_LENGTH)
    atr = _atr(h, l_arr, c, ATR_PERIOD)
    ax = adx(h, l_arr, c, ADX_LENGTH)
    cx = cci(h, l_arr, c, CCI_LENGTH, CCI_SOURCE)

    event_start_ts = pd.Timestamp(event_start, tz="UTC")
    rows: list[dict] = []

    # Start after enough warmup for indicators.
    start_i = max(DEMA_LENGTH + 50, 200)
    for i in range(start_i, len(df_bars)):
        if any(np.isnan(x[i]) for x in (st, atr, d100, d200)):
            continue
        event = detect_pullback_event(
            open_=float(o[i]),
            high=float(h[i]),
            low=float(l_arr[i]),
            close=float(c[i]),
            st=float(st[i]),
            st_dir=int(direction[i]),
            prev_st_dir=int(direction[i - 1]),
            atr=float(atr[i]),
            dema_100=float(d100[i]),
            dema_200=float(d200[i]),
        )
        if event is None:
            continue

        entry_ts = df_bars.index[i]
        if entry_ts < event_start_ts:
            continue

        # Simulate forward: SL, TP, or 100-bar timeout.
        end_i = min(i + 1 + AGGR_MAX_HOLD_BARS, len(df_bars))
        future_h = h[i + 1:end_i]
        future_l = l_arr[i + 1:end_i]
        future_c = c[i + 1:end_i]

        if event.side == "Long":
            sl_hit = np.where(future_l <= event.sl_price)[0]
            tp_hit = np.where(future_h >= event.tp_price)[0]
        else:
            sl_hit = np.where(future_h >= event.sl_price)[0]
            tp_hit = np.where(future_l <= event.tp_price)[0]

        first_sl = int(sl_hit[0]) if len(sl_hit) else None
        first_tp = int(tp_hit[0]) if len(tp_hit) else None

        if first_tp is not None and (first_sl is None or first_tp < first_sl):
            exit_idx = first_tp
            exit_price = event.tp_price
            exit_reason = "TP"
            pnl_pts = abs(event.tp_price - event.entry_price)
        elif first_sl is not None:
            exit_idx = first_sl
            exit_price = event.sl_price
            exit_reason = "SL"
            pnl_pts = -abs(event.entry_price - event.sl_price)
        elif len(future_c):
            exit_idx = len(future_c) - 1
            exit_price = float(future_c[-1])
            exit_reason = "TIMEOUT"
            pnl_pts = (exit_price - event.entry_price) if event.side == "Long" \
                      else (event.entry_price - exit_price)
        else:
            continue  # no forward bars

        exit_i = i + 1 + exit_idx
        exit_ts = df_bars.index[exit_i]
        bars_held = exit_i - i
        risk_points = event.risk_pts
        pnl_usd = pnl_pts * POINT_VALUE_USD - ROUND_TURN_COMMISSION_USD
        r_multiple = pnl_pts / risk_points if risk_points > 0 else 0.0

        rows.append({
            "side": event.side,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "entry_price": event.entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "duration_min": bars_held * timeframe_min,
            "gross_points": pnl_pts,
            "gross_usd": pnl_pts * POINT_VALUE_USD,
            "commission_usd": ROUND_TURN_COMMISSION_USD,
            "pnl_usd": pnl_usd,
            "risk_points": risk_points,
            "risk_usd": risk_points * POINT_VALUE_USD,
            "r_multiple": r_multiple,
            "mfe_points": 0.0,
            "mae_points": 0.0,
            "mfe_usd": 0.0,
            "mae_usd": 0.0,
            "entry_adx": float(ax[i]) if not np.isnan(ax[i]) else 0.0,
            "entry_cci": float(cx[i]) if not np.isnan(cx[i]) else 0.0,
            "entry_dema": float(d200[i]),
            "entry_supertrend": float(st[i]),
            "entry_st_direction": int(direction[i]),
            "entry_atr": float(atr[i]),
            "entry_bar_high": float(h[i]),
            "entry_bar_low": float(l_arr[i]),
            "entry_bar_close": float(c[i]),
            "hour_utc": entry_ts.hour,
            "minute_utc": entry_ts.minute,
            "session": session_name(entry_ts),
            "is_win": pnl_usd > 0,
            "hit_1r": pnl_pts >= risk_points if risk_points > 0 else False,
            "hit_2r": False,
            "entry_i": i,
            "exit_i": exit_i,
            "mode": "AGGR",
        })

    return pd.DataFrame(rows)


def build_events(df_bars: pd.DataFrame, event_start: str, timeframe_min: int) -> pd.DataFrame:
    h = df_bars["high"].to_numpy(dtype=float)
    l = df_bars["low"].to_numpy(dtype=float)
    c = df_bars["close"].to_numpy(dtype=float)

    st, direction = supertrend(h, l, c, ST_FACTOR, ATR_PERIOD)
    d = dema(c, DEMA_LENGTH)
    ax = adx(h, l, c, ADX_LENGTH)
    cx = cci(h, l, c, CCI_LENGTH, CCI_SOURCE)
    atr = _atr(h, l, c, ATR_PERIOD)

    event_start_ts = pd.Timestamp(event_start, tz="UTC")
    pos = 0
    open_trade: OpenTrade | None = None
    sl_price = 0.0
    rows: list[dict] = []

    for i in range(DEMA_LENGTH + 50, len(df_bars)):
        ts = df_bars.index[i]
        cur_close = float(c[i])
        cur_high = float(h[i])
        cur_low = float(l[i])
        cur_dema = float(d[i])
        cur_st = float(st[i]) if not np.isnan(st[i]) else np.nan
        cur_dir = int(direction[i])
        cur_adx = float(ax[i]) if not np.isnan(ax[i]) else 0.0
        cur_cci = float(cx[i]) if not np.isnan(cx[i]) else 0.0
        cur_atr = float(atr[i]) if not np.isnan(atr[i]) else np.nan

        cross_up = float(c[i - 1]) < d[i - 1] and cur_close > cur_dema
        cross_dn = float(c[i - 1]) > d[i - 1] and cur_close < cur_dema
        long_signal = (
            cur_adx > ADX_THRESHOLD
            and cur_cci > CCI_LONG_MIN
            and (cross_up or cur_close > cur_dema)
            and cur_dir < 0
        )
        short_signal = (
            cur_adx > ADX_THRESHOLD
            and cur_cci < CCI_SHORT_MAX
            and (cross_dn or cur_close < cur_dema)
            and cur_dir > 0
        )

        if open_trade is not None:
            update_excursions(open_trade, cur_high, cur_low)

        if pos == 1 and open_trade is not None and cur_low <= sl_price:
            if open_trade.entry_ts >= event_start_ts:
                rows.append(close_trade(open_trade, ts, i, float(sl_price), "SL", i - open_trade.entry_i, timeframe_min))
            pos = 0
            open_trade = None
        elif pos == -1 and open_trade is not None and cur_high >= sl_price:
            if open_trade.entry_ts >= event_start_ts:
                rows.append(close_trade(open_trade, ts, i, float(sl_price), "SL", i - open_trade.entry_i, timeframe_min))
            pos = 0
            open_trade = None

        if pos == 1 and open_trade is not None and cur_dir > 0:
            if open_trade.entry_ts >= event_start_ts:
                rows.append(close_trade(open_trade, ts, i, cur_close, "TREND_FLIP", i - open_trade.entry_i, timeframe_min))
            pos = 0
            open_trade = None
        elif pos == -1 and open_trade is not None and cur_dir < 0:
            if open_trade.entry_ts >= event_start_ts:
                rows.append(close_trade(open_trade, ts, i, cur_close, "TREND_FLIP", i - open_trade.entry_i, timeframe_min))
            pos = 0
            open_trade = None

        if pos != 0:
            sl_price = cur_st

        if long_signal and pos == 0:
            pos = 1
            sl_price = cur_st
            open_trade = OpenTrade(
                side="Long",
                entry_ts=ts,
                entry_i=i,
                entry_price=cur_close,
                initial_sl=cur_st,
                sl_price=cur_st,
                entry_adx=cur_adx,
                entry_cci=cur_cci,
                entry_dema=cur_dema,
                entry_supertrend=cur_st,
                entry_direction=cur_dir,
                entry_atr=cur_atr,
                entry_bar_high=cur_high,
                entry_bar_low=cur_low,
                entry_bar_close=cur_close,
            )
        elif short_signal and pos == 0:
            pos = -1
            sl_price = cur_st
            open_trade = OpenTrade(
                side="Short",
                entry_ts=ts,
                entry_i=i,
                entry_price=cur_close,
                initial_sl=cur_st,
                sl_price=cur_st,
                entry_adx=cur_adx,
                entry_cci=cur_cci,
                entry_dema=cur_dema,
                entry_supertrend=cur_st,
                entry_direction=cur_dir,
                entry_atr=cur_atr,
                entry_bar_high=cur_high,
                entry_bar_low=cur_low,
                entry_bar_close=cur_close,
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["entry_date"] = out["entry_ts"].dt.date.astype(str)
        out["exit_date"] = out["exit_ts"].dt.date.astype(str)
        out["trade_no"] = np.arange(1, len(out) + 1)
    if open_trade is not None and open_trade.entry_ts >= event_start_ts:
        last_ts = df_bars.index[-1]
        last_i = len(df_bars) - 1
        last_close = float(c[-1])
        gross_points = (
            last_close - open_trade.entry_price
            if open_trade.side == "Long"
            else open_trade.entry_price - last_close
        )
        risk_points = abs(open_trade.entry_price - open_trade.initial_sl)
        out.attrs["open_trade"] = {
            "entry_ts": open_trade.entry_ts.strftime("%Y-%m-%d %H:%M"),
            "entry_time": int(open_trade.entry_ts.timestamp()),
            "last_ts": last_ts.strftime("%Y-%m-%d %H:%M"),
            "last_time": int(last_ts.timestamp()),
            "side": open_trade.side,
            "entry_price": open_trade.entry_price,
            "last_price": last_close,
            "unrealized_pnl_usd": gross_points * POINT_VALUE_USD,
            "risk_points": risk_points,
            "entry_adx": open_trade.entry_adx,
            "entry_cci": open_trade.entry_cci,
            "entry_dema": open_trade.entry_dema,
            "entry_supertrend": open_trade.entry_supertrend,
            "entry_st_direction": open_trade.entry_direction,
            "entry_i": open_trade.entry_i,
            "last_i": last_i,
            "duration_min": (last_i - open_trade.entry_i) * timeframe_min,
            "session": session_name(open_trade.entry_ts),
        }
    return out


def candle_records(df: pd.DataFrame, start: str, end: str) -> list[dict]:
    view = df.loc[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))].copy()
    return [
        {
            "time": int(ts.timestamp()),
            "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "open": round(float(row.open), 4),
            "high": round(float(row.high), 4),
            "low": round(float(row.low), 4),
            "close": round(float(row.close), 4),
            "volume": round(float(row.volume), 2),
        }
        for ts, row in view.iterrows()
    ]


def trade_records(events: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in events.iterrows():
        d = row.to_dict()
        dema_d = d.get("dema_distance_atr")
        st_d = d.get("st_distance_atr")
        records.append({
            "trade_no": int(row.trade_no),
            "entry_time": int(row.entry_ts.timestamp()),
            "exit_time": int(row.exit_ts.timestamp()),
            "entry_ts": row.entry_ts.strftime("%Y-%m-%d %H:%M"),
            "exit_ts": row.exit_ts.strftime("%Y-%m-%d %H:%M"),
            "side": row.side,
            "mode": d.get("mode", "CONS"),
            "entry_price": round(float(row.entry_price), 4),
            "exit_price": round(float(row.exit_price), 4),
            "exit_reason": row.exit_reason,
            "pnl_usd": round(float(row.pnl_usd), 2),
            "r_multiple": round(float(row.r_multiple), 4) if pd.notna(row.r_multiple) else None,
            "mfe_usd": round(float(row.mfe_usd), 2),
            "mae_usd": round(float(row.mae_usd), 2),
            "entry_adx": round(float(row.entry_adx), 2),
            "entry_cci": round(float(row.entry_cci), 2),
            "dema_distance_atr": round(float(dema_d), 4) if pd.notna(dema_d) else None,
            "st_distance_atr": round(float(st_d), 4) if pd.notna(st_d) else None,
            "session": row.session,
            "is_win": bool(row.is_win),
            "hit_1r": bool(row.hit_1r),
            "hit_2r": bool(row.hit_2r),
            "duration_min": int(row.duration_min),
        })
    return records


def open_trade_records(events: pd.DataFrame) -> list[dict]:
    open_trade = events.attrs.get("open_trade")
    if not open_trade:
        return []
    return [{
        "status": "OPEN",
        "entry_time": int(open_trade["entry_time"]),
        "exit_time": None,
        "entry_ts": open_trade["entry_ts"],
        "exit_ts": "",
        "last_ts": open_trade["last_ts"],
        "last_time": int(open_trade["last_time"]),
        "side": open_trade["side"],
        "entry_price": round(float(open_trade["entry_price"]), 4),
        "last_price": round(float(open_trade["last_price"]), 4),
        "exit_price": None,
        "exit_reason": "OPEN",
        "unrealized_pnl_usd": round(float(open_trade["unrealized_pnl_usd"]), 2),
        "risk_points": round(float(open_trade["risk_points"]), 4),
        "entry_adx": round(float(open_trade["entry_adx"]), 2),
        "entry_cci": round(float(open_trade["entry_cci"]), 2),
        "session": open_trade["session"],
        "duration_min": int(open_trade["duration_min"]),
    }]


def summary_record(events: pd.DataFrame, start: str, end: str, timeframe: str, candle_count: int) -> dict:
    gross_profit = float(events.loc[events["pnl_usd"] > 0, "pnl_usd"].sum()) if not events.empty else 0.0
    gross_loss = abs(float(events.loc[events["pnl_usd"] < 0, "pnl_usd"].sum())) if not events.empty else 0.0
    return {
        "start": start,
        "end": end,
        "timeframe": timeframe,
        "candles": candle_count,
        "trades": len(events),
        "total_pnl_usd": round(float(events["pnl_usd"].sum()), 2) if not events.empty else 0.0,
        "win_rate": round(float(events["is_win"].mean()), 4) if not events.empty else 0.0,
        "avg_r": round(float(events["r_multiple"].mean()), 4) if not events.empty else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
    }


def export_ui(bars_by_tf: dict[str, pd.DataFrame], events_by_tf: dict[str, pd.DataFrame], start: str, end: str) -> None:
    OUT_CANDLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    candle_sets = {
        timeframe: candle_records(df_bars, start, end)
        for timeframe, df_bars in bars_by_tf.items()
    }

    for timeframe, candles in candle_sets.items():
        summary = summary_record(events_by_tf[timeframe], start, end, timeframe, len(candles))
        out_path = ROOT / f"ui/data/candles_{STRATEGY_KEY}_{timeframe}.json"
        out_path.write_text(json.dumps({"summary": summary, "candles": candles}, separators=(",", ":")))

        trades_path = ROOT / f"ui/data/trade_events_{STRATEGY_KEY}_{timeframe}.json"
        trades_path.write_text(json.dumps({
            "summary": summary,
            "trades": trade_records(events_by_tf[timeframe]),
            "open_trades": open_trade_records(events_by_tf[timeframe]),
        }, separators=(",", ":")))

    # Backward compat alias (legacy consumers)
    OUT_TRADES_JSON.write_text((ROOT / f"ui/data/trade_events_{STRATEGY_KEY}_5m.json").read_text())


def main() -> None:
    args = parse_args()
    print(f"Loading 1m OHLCV from {args.raw_db}...")
    df_1m = load_ohlcv_1m(args.raw_db, args.start, args.end, args.table)
    bars_by_tf = {
        "1m": df_1m.set_index("timestamp_utc").sort_index(),
        "5m": resample_5m(df_1m),
        "15m": resample_ohlcv(df_1m, "15min"),
    }
    print(
        f"Loaded {len(df_1m):,} 1m rows -> "
        f"{len(bars_by_tf['5m']):,} 5m candles / {len(bars_by_tf['15m']):,} 15m candles"
    )

    events_by_tf = {
        "1m": build_events(bars_by_tf["1m"], args.start, 1),
        "5m": build_events(bars_by_tf["5m"], args.start, 5),
        "15m": build_events(bars_by_tf["15m"], args.start, 15),
    }
    # Tag every existing trade as CONS (DEMA cross + ML refined gate in live).
    for tf, df_ev in events_by_tf.items():
        if not df_ev.empty and "mode" not in df_ev.columns:
            df_ev["mode"] = "CONS"
    # AGGR (v1.12 pullback) overlay — only meaningful on 5m bars (matches live).
    aggr_5m = build_aggr_pullback_events(bars_by_tf["5m"], args.start, 5)
    if not aggr_5m.empty:
        events_by_tf["5m"] = (
            pd.concat([events_by_tf["5m"], aggr_5m], ignore_index=True)
            .sort_values("entry_ts")
            .reset_index(drop=True)
        )
        # Re-number trade_no globally after concat (AGGR rows had no trade_no).
        events_by_tf["5m"]["trade_no"] = np.arange(1, len(events_by_tf["5m"]) + 1)
        # Ensure derived date columns exist for AGGR rows too.
        events_by_tf["5m"]["entry_date"] = events_by_tf["5m"]["entry_ts"].dt.date.astype(str)
        events_by_tf["5m"]["exit_date"] = events_by_tf["5m"]["exit_ts"].dt.date.astype(str)
        print(f"AGGR pullback overlay: {len(aggr_5m):,} events added to 5m")
    events = events_by_tf["5m"].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.out, index=False)
    print(f"Saved {len(events):,} trade events -> {args.out}")

    for timeframe, tf_events in events_by_tf.items():
        if tf_events.empty:
            print(f"{timeframe}: no trades")
            continue
        print(
            f"{timeframe}: trades={len(tf_events):,}, "
            f"pnl=${tf_events['pnl_usd'].sum():.2f}, "
            f"win_rate={tf_events['is_win'].mean() * 100:.1f}%, "
            f"avg_R={tf_events['r_multiple'].mean():.3f}"
        )

    if args.export_ui:
        export_ui(bars_by_tf, events_by_tf, args.start, args.end)
        print(f"Saved UI candles -> {OUT_CANDLES_JSON.parent}/candles_{{1m,5m,15m}}.json")
        print(f"Saved UI trades -> {OUT_TRADES_JSON.parent}/trade_events_{{1m,5m,15m}}.json")


if __name__ == "__main__":
    main()
