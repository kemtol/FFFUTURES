#!/usr/bin/env python3
"""Build trade-event datamart for FVG + DEMA Scalper strategy.

Mirrors build_st_trade_events.py with FVG detection, DEMA filter, ADX+CHOP
regime, session gating, and swing SL/TP.
"""
from __future__ import annotations

import argparse, json, sqlite3, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.fvg_scalper import (
    dema, atr, adx, choppiness,
    MIN_GAP_PTS, MIN_GAP_ATR, USE_DISPL, DISPL_ATR, MIN_BODY_PCT, FVG_EXTEND_BARS,
    DEMA_LENGTH, USE_DEMA_DIR, USE_DEMA_SLOPE, SLOPE_BARS,
    USE_DIST_FILTER, MAX_DIST_ATR,
    USE_REGIME, ADX_LENGTH, MIN_ADX, CHOP_LENGTH, MAX_CHOP,
    MIN_DEMA_SLOPE_ATR, USE_WHIPSAW, CROSS_LOOKBACK, MAX_DEMA_CROSSES,
    TP_RISK_RATIO, SL_LOOKBACK, MIN_RISK_PTS, MAX_RISK_PTS,
    MAX_TRADES_DAY, COOLDOWN_BARS, ONE_TRADE_AT_TIME,
    USE_DEMA_EXIT, DEMA_EXIT_BUFFER, DEMA_EXIT_ONLY_LOSS,
    ENABLE_LONG, ENABLE_SHORT, SESSION_MODE,
    COMMISSION, POINT_VALUE,
)

RAW_DB = ROOT / "data/Level_0_Raw/MGC_1m.db"
OUT_PARQUET = ROOT / "data/Level_2_Datamart/fvg_scalper_trade_events.parquet"
OUT_CANDLES_JSON = ROOT / "ui/data/candles_fvg_scalper_5m.json"
OUT_TRADES_JSON = ROOT / "ui/data/trade_events_fvg_scalper.json"
STRATEGY_KEY = "fvg_scalper"

SYMBOL = "MICRO_GOLD"
TIMEFRAME = "1m"
WARMUP_DAYS = 120


@dataclass
class OpenTrade:
    side: str
    entry_ts: pd.Timestamp
    entry_i: int
    entry_price: float
    initial_sl: float
    sl_price: float
    tp_price: float
    entry_adx: float
    entry_chop: float
    entry_dema: float
    entry_dema_slope: float
    entry_gap_pts: float
    entry_atr: float
    entry_bar_high: float
    entry_bar_low: float
    entry_bar_close: float
    mfe_points: float = 0.0
    mae_points: float = 0.0


def session_name(ts: pd.Timestamp) -> str:
    t = ts.tz_convert("UTC") if ts.tz else ts.tz_localize("UTC")
    h = t.hour + t.minute / 60.0
    if 0 <= h < 7: return "Asia"
    if 7 <= h < 12: return "London"
    if 13.5 <= h < 20: return "NY"
    return "Other"


def is_in_session(ts: pd.Timestamp, mode: str = "Off") -> bool:
    if mode == "Off": return True
    t = ts.tz_convert("UTC") if ts.tz else ts.tz_localize("UTC")
    h = t.hour + t.minute / 60.0
    in_asia = 0 <= h < 7
    in_london = 7 <= h < 12
    in_ny = 13.5 <= h < 20
    if mode == "Asia Only": return in_asia
    if mode == "London Only": return in_london
    if mode == "NY Only": return in_ny
    if mode == "Asia + London": return in_asia or in_london
    if mode == "Asia + NY": return in_asia or in_ny
    if mode == "London + NY": return in_london or in_ny
    if mode == "Asia + London + NY": return in_asia or in_london or in_ny
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2026-01-01", help="event start date UTC")
    p.add_argument("--end", default="2026-05-01", help="event end date UTC, exclusive")
    p.add_argument("--raw-db", type=Path, default=RAW_DB)
    p.add_argument("--table", default="investing_ohlcv_1m", help="Source table name")
    p.add_argument("--out", type=Path, default=OUT_PARQUET)
    p.add_argument("--export-ui", action="store_true", help="write ui/data JSON files")
    return p.parse_args()


def load_ohlcv_1m(db_path: Path, start: str, end: str, table: str = "investing_ohlcv_1m") -> pd.DataFrame:
    ws = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS))
    warmup_start = ws.strftime("%Y-%m-%d %H:%M:%S")
    end_ts = pd.Timestamp(end, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql(
            f"""SELECT timestamp_utc, open, high, low, close, volume
               FROM {table}
               WHERE symbol=? AND timeframe=?
                 AND timestamp_utc >= ? AND timestamp_utc < ?
               ORDER BY epoch_ms""",
            conn, params=[SYMBOL, TIMEFRAME, warmup_start, end_ts],
        )
    if df.empty:
        raise RuntimeError(f"No OHLCV rows found for {warmup_start} -> {end_ts}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df_1m.set_index("timestamp_utc").sort_index()
    return df.resample(rule, label="right", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlcv(df_1m, "5min")


def close_trade(trade: OpenTrade, exit_ts: pd.Timestamp, exit_i: int,
                exit_price: float, exit_reason: str,
                bars_held: int, timeframe_min: int) -> dict:
    gross_pts = exit_price - trade.entry_price if trade.side == "Long" else trade.entry_price - exit_price
    gross_usd = gross_pts * POINT_VALUE
    pnl_usd = gross_usd - COMMISSION
    risk_pts = abs(trade.entry_price - trade.initial_sl)
    risk_usd = risk_pts * POINT_VALUE
    r_mult = gross_usd / risk_usd if risk_usd > 0 else np.nan

    return {
        "entry_ts": trade.entry_ts, "exit_ts": exit_ts,
        "side": trade.side,
        "entry_price": trade.entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "duration_min": bars_held * timeframe_min,
        "gross_points": gross_pts, "gross_usd": gross_usd,
        "commission_usd": COMMISSION, "pnl_usd": pnl_usd,
        "risk_points": risk_pts, "risk_usd": risk_usd,
        "r_multiple": r_mult,
        "mfe_points": trade.mfe_points, "mae_points": trade.mae_points,
        "mfe_usd": trade.mfe_points * POINT_VALUE,
        "mae_usd": trade.mae_points * POINT_VALUE,
        "entry_adx": trade.entry_adx, "entry_chop": trade.entry_chop,
        "entry_dema": trade.entry_dema, "entry_dema_slope": trade.entry_dema_slope,
        "entry_gap_pts": trade.entry_gap_pts, "entry_atr": trade.entry_atr,
        "entry_bar_high": trade.entry_bar_high,
        "entry_bar_low": trade.entry_bar_low,
        "entry_bar_close": trade.entry_bar_close,
        "dema_distance": (trade.entry_price - trade.entry_dema
                          if trade.side == "Long" else trade.entry_dema - trade.entry_price),
        "dema_distance_atr": (abs(trade.entry_price - trade.entry_dema)
                              / trade.entry_atr if trade.entry_atr > 0 else np.nan),
        "hour_utc": trade.entry_ts.hour, "minute_utc": trade.entry_ts.minute,
        "session": session_name(trade.entry_ts),
        "is_win": pnl_usd > 0,
        "hit_1r": trade.mfe_points >= risk_pts if risk_pts > 0 else False,
        "hit_2r": trade.mfe_points >= 2 * risk_pts if risk_pts > 0 else False,
        "entry_i": trade.entry_i, "exit_i": exit_i,
    }


def build_events(df_bars: pd.DataFrame, event_start: str,
                 timeframe_min: int, params: dict | None = None) -> pd.DataFrame:
    p = params or {}
    _min_gap = p.get("MIN_GAP_PTS", MIN_GAP_PTS)
    _use_displ = p.get("USE_DISPL", USE_DISPL)
    _displ_atr = p.get("DISPL_ATR", DISPL_ATR)
    _min_body = p.get("MIN_BODY_PCT", MIN_BODY_PCT)
    _use_dema_dir = p.get("USE_DEMA_DIR", USE_DEMA_DIR)
    _use_dema_slope = p.get("USE_DEMA_SLOPE", USE_DEMA_SLOPE)
    _slope_bars = p.get("SLOPE_BARS", SLOPE_BARS)
    _use_dist = p.get("USE_DIST_FILTER", USE_DIST_FILTER)
    _max_dist = p.get("MAX_DIST_ATR", MAX_DIST_ATR)
    _use_regime = p.get("USE_REGIME", USE_REGIME)
    _min_adx = p.get("MIN_ADX", MIN_ADX)
    _max_chop = p.get("MAX_CHOP", MAX_CHOP)
    _min_slope_atr = p.get("MIN_DEMA_SLOPE_ATR", MIN_DEMA_SLOPE_ATR)
    _tp_ratio = p.get("TP_RISK_RATIO", TP_RISK_RATIO)
    _sl_lookback = p.get("SL_LOOKBACK", SL_LOOKBACK)
    _min_risk = p.get("MIN_RISK_PTS", MIN_RISK_PTS)
    _max_risk = p.get("MAX_RISK_PTS", MAX_RISK_PTS)
    _max_trades = p.get("MAX_TRADES_DAY", MAX_TRADES_DAY)
    _cooldown = p.get("COOLDOWN_BARS", COOLDOWN_BARS)
    _one_trade = p.get("ONE_TRADE_AT_TIME", ONE_TRADE_AT_TIME)
    _enable_long = p.get("ENABLE_LONG", ENABLE_LONG)
    _enable_short = p.get("ENABLE_SHORT", ENABLE_SHORT)
    _session_mode = p.get("SESSION_MODE", SESSION_MODE)
    _use_dema_exit = p.get("USE_DEMA_EXIT", USE_DEMA_EXIT)
    _dema_exit_buf = p.get("DEMA_EXIT_BUFFER", DEMA_EXIT_BUFFER)
    _dema_exit_loss = p.get("DEMA_EXIT_ONLY_LOSS", DEMA_EXIT_ONLY_LOSS)
    _comm = p.get("COMMISSION", COMMISSION)
    _pt_val = p.get("POINT_VALUE", POINT_VALUE)
    h = df_bars["high"].to_numpy(dtype=float)
    l = df_bars["low"].to_numpy(dtype=float)
    c = df_bars["close"].to_numpy(dtype=float)
    o = df_bars["open"].to_numpy(dtype=float)

    hl2 = (h + l) / 2.0
    d_arr = dema(hl2, DEMA_LENGTH)
    ax = adx(h, l, c, ADX_LENGTH)
    atr_arr = atr(h, l, c, 14)
    chop_arr = choppiness(h, l, c, CHOP_LENGTH)

    event_start_ts = pd.Timestamp(event_start, tz="UTC")
    pos = 0
    open_trade: OpenTrade | None = None
    sl_price = 0.0
    tp_price = 0.0
    entry_price = 0.0
    trades_today = 0
    last_date = None
    last_entry_i = -100
    rows: list[dict] = []

    for i in range(DEMA_LENGTH + 10, len(c)):
        cur_close = float(c[i])
        cur_dema = float(d_arr[i]) if i < len(d_arr) and not np.isnan(d_arr[i]) else 0.0
        cur_adx = float(ax[i]) if i < len(ax) and not np.isnan(ax[i]) else 0.0
        cur_chop = float(chop_arr[i]) if i < len(chop_arr) and not np.isnan(chop_arr[i]) else 100.0
        cur_atr = float(atr_arr[i]) if i < len(atr_arr) and not np.isnan(atr_arr[i]) else 1.0
        sl_b = min(_slope_bars, i)
        cur_slope = cur_dema - float(d_arr[i - sl_b]) if i >= sl_b and not np.isnan(d_arr[i - sl_b]) else 0.0
        cur_ts = df_bars.index[i]
        cur_high = float(h[i])
        cur_low = float(l[i])

        # daily counter
        bar_date = cur_ts.date()
        if last_date is None or bar_date != last_date:
            trades_today = 0
            last_date = bar_date

        # DEMA filter
        distance = abs(cur_close - cur_dema)
        above_dema = cur_close > cur_dema
        below_dema = cur_close < cur_dema
        slope_up = cur_slope > 0
        slope_down = cur_slope < 0
        dist_ok = not _use_dist or distance <= cur_atr * _max_dist
        long_trend_ok = not _use_dema_dir or above_dema
        short_trend_ok = not _use_dema_dir or below_dema
        long_slope_ok = not _use_dema_slope or slope_up
        short_slope_ok = not _use_dema_slope or slope_down

        # regime
        adx_ok = cur_adx >= _min_adx
        chop_ok = cur_chop <= _max_chop
        slope_strength = abs(cur_slope) / cur_atr if cur_atr > 0 else 0.0
        slope_ok = slope_strength >= _min_slope_atr
        regime_ok = not _use_regime or (adx_ok and chop_ok and slope_ok)

        # session
        in_session = is_in_session(cur_ts, _session_mode)

        # trade mgmt
        daily_ok = trades_today < _max_trades
        cooldown_ok = i - last_entry_i >= _cooldown
        flat_ok = not _one_trade or pos == 0

        # FVG detection
        fvg = None
        fvg_gap = 0.0
        if i >= 2:
            h2, l2 = float(h[i - 2]), float(l[i - 2])
            h1, l1 = float(h[i - 1]), float(l[i - 1])
            c1 = float(c[i - 1])
            o1 = float(o[i - 1])

            bull_raw = l[i] > h2 and c1 > h2
            bear_raw = h[i] < l2 and c1 < l2
            bull_gap = l[i] - h2 if bull_raw else 0.0
            bear_gap = l2 - h[i] if bear_raw else 0.0
            gap_pts = bull_gap if bull_raw else bear_gap

            gap_atr_ok = gap_pts >= cur_atr * MIN_GAP_ATR
            gap_pts_ok = gap_pts >= _min_gap
            mid_range = h1 - l1
            mid_body = abs(c1 - o1)
            mid_body_pct = mid_body / mid_range * 100.0 if mid_range > 0 else 0.0

            bull_disp_ok = (not _use_displ or
                            (c1 > o1 and mid_range >= cur_atr * _displ_atr and
                             mid_body_pct >= _min_body))
            bear_disp_ok = (not _use_displ or
                            (c1 < o1 and mid_range >= cur_atr * _displ_atr and
                             mid_body_pct >= _min_body))

            bull_fvg = bull_raw and gap_pts_ok and gap_atr_ok and bull_disp_ok
            bear_fvg = bear_raw and gap_pts_ok and gap_atr_ok and bear_disp_ok
            fvg = "bull" if bull_fvg else "bear" if bear_fvg else None
            fvg_gap = gap_pts

        # risk calc
        sl_bars = min(_sl_lookback - 1, i)
        long_sl_base = float(np.min(l[i - sl_bars:i + 1]))
        short_sl_base = float(np.max(h[i - sl_bars:i + 1]))
        long_risk = cur_close - long_sl_base
        short_risk = short_sl_base - cur_close
        long_risk_ok = _min_risk <= long_risk <= _max_risk
        short_risk_ok = _min_risk <= short_risk <= _max_risk

        # entries
        long_signal = (_enable_long and fvg == "bull" and long_trend_ok and
                       long_slope_ok and dist_ok and regime_ok and
                       in_session and daily_ok and cooldown_ok and flat_ok and
                       long_risk_ok)
        short_signal = (_enable_short and fvg == "bear" and short_trend_ok and
                        short_slope_ok and dist_ok and regime_ok and
                        in_session and daily_ok and cooldown_ok and flat_ok and
                        short_risk_ok)

        # exits
        if pos == 1 and open_trade is not None:
            if cur_low <= sl_price:
                if open_trade.entry_ts >= event_start_ts:
                    rows.append(close_trade(open_trade, cur_ts, i, sl_price,
                                            "SL", i - open_trade.entry_i,
                                            timeframe_min))
                pos = 0; open_trade = None
            elif cur_high >= tp_price:
                if open_trade.entry_ts >= event_start_ts:
                    rows.append(close_trade(open_trade, cur_ts, i, tp_price,
                                            "TP", i - open_trade.entry_i,
                                            timeframe_min))
                pos = 0; open_trade = None
            elif _use_dema_exit:
                prev_c = float(c[i - 1])
                if prev_c > cur_dema and cur_close < cur_dema:
                    if not _dema_exit_loss or cur_close < entry_price:
                        if open_trade.entry_ts >= event_start_ts:
                            rows.append(close_trade(open_trade, cur_ts, i,
                                                    cur_close, "DEMA_EXIT",
                                                    i - open_trade.entry_i,
                                                    timeframe_min))
                        pos = 0; open_trade = None
            if open_trade is not None:
                mfe = (cur_high - entry_price) if open_trade.side == "Long" else (entry_price - cur_low)
                mae = (cur_low - entry_price) if open_trade.side == "Long" else (entry_price - cur_high)
                open_trade.mfe_points = max(open_trade.mfe_points, mfe)
                open_trade.mae_points = min(open_trade.mae_points, mae)

        if pos == -1 and open_trade is not None:
            if cur_high >= sl_price:
                if open_trade.entry_ts >= event_start_ts:
                    rows.append(close_trade(open_trade, cur_ts, i, sl_price,
                                            "SL", i - open_trade.entry_i,
                                            timeframe_min))
                pos = 0; open_trade = None
            elif cur_low <= tp_price:
                if open_trade.entry_ts >= event_start_ts:
                    rows.append(close_trade(open_trade, cur_ts, i, tp_price,
                                            "TP", i - open_trade.entry_i,
                                            timeframe_min))
                pos = 0; open_trade = None
            elif _use_dema_exit:
                prev_c = float(c[i - 1])
                if prev_c < cur_dema and cur_close > cur_dema:
                    if not _dema_exit_loss or cur_close > entry_price:
                        if open_trade.entry_ts >= event_start_ts:
                            rows.append(close_trade(open_trade, cur_ts, i,
                                                    cur_close, "DEMA_EXIT",
                                                    i - open_trade.entry_i,
                                                    timeframe_min))
                        pos = 0; open_trade = None
            if open_trade is not None:
                mfe = (entry_price - cur_low) if open_trade.side == "Short" else (cur_high - entry_price)
                mae = (entry_price - cur_high) if open_trade.side == "Short" else (cur_low - entry_price)
                open_trade.mfe_points = max(open_trade.mfe_points, mfe)
                open_trade.mae_points = min(open_trade.mae_points, mae)

        # entries
        if long_signal and pos == 0:
            pos = 1
            sl_price = long_sl_base
            tp_price = cur_close + _tp_ratio * long_risk
            entry_price = cur_close
            trades_today += 1
            last_entry_i = i
            dema_slope = cur_slope / cur_atr if cur_atr > 0 else 0
            open_trade = OpenTrade(
                side="Long", entry_ts=cur_ts, entry_i=i,
                entry_price=cur_close, initial_sl=long_sl_base,
                sl_price=sl_price, tp_price=tp_price,
                entry_adx=cur_adx, entry_chop=cur_chop,
                entry_dema=cur_dema, entry_dema_slope=dema_slope,
                entry_gap_pts=fvg_gap, entry_atr=cur_atr,
                entry_bar_high=cur_high, entry_bar_low=cur_low,
                entry_bar_close=cur_close)

        if short_signal and pos == 0:
            pos = -1
            sl_price = short_sl_base
            tp_price = cur_close - _tp_ratio * short_risk
            entry_price = cur_close
            trades_today += 1
            last_entry_i = i
            dema_slope = cur_slope / cur_atr if cur_atr > 0 else 0
            open_trade = OpenTrade(
                side="Short", entry_ts=cur_ts, entry_i=i,
                entry_price=cur_close, initial_sl=short_sl_base,
                sl_price=sl_price, tp_price=tp_price,
                entry_adx=cur_adx, entry_chop=cur_chop,
                entry_dema=cur_dema, entry_dema_slope=dema_slope,
                entry_gap_pts=fvg_gap, entry_atr=cur_atr,
                entry_bar_high=cur_high, entry_bar_low=cur_low,
                entry_bar_close=cur_close)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["entry_date"] = out["entry_ts"].dt.date.astype(str)
        out["exit_date"] = out["exit_ts"].dt.date.astype(str)
        out["trade_no"] = np.arange(1, len(out) + 1)
    return out


# ── UI export ────────────────────────────────────────────────────────────

def candle_records(df: pd.DataFrame, start: str, end: str) -> list[dict]:
    view = df.loc[(df.index >= pd.Timestamp(start, tz="UTC")) &
                  (df.index < pd.Timestamp(end, tz="UTC"))].copy()
    return [
        {"time": int(ts.timestamp()),
         "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
         "open": round(float(row.open), 4),
         "high": round(float(row.high), 4),
         "low": round(float(row.low), 4),
         "close": round(float(row.close), 4),
         "volume": int(getattr(row, "volume", 0))}
        for ts, row in view.iterrows()
    ]


def trade_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        out.append({
            "trade_no": int(r.trade_no),
            "entry_time": int(r.entry_ts.timestamp()),
            "exit_time": int(r.exit_ts.timestamp()),
            "entry_ts": str(r.entry_ts)[:16],
            "exit_ts": str(r.exit_ts)[:16],
            "side": str(r.side),
            "entry_price": round(float(r.entry_price), 1),
            "exit_price": round(float(r.exit_price), 4),
            "exit_reason": str(r.exit_reason),
            "pnl_usd": float(r.pnl_usd),
            "r_multiple": round(float(r.r_multiple), 4) if not np.isnan(float(r.r_multiple)) else None,
            "entry_adx": round(float(r.entry_adx), 2),
            "entry_chop": round(float(r.entry_chop), 2),
            "entry_dema": round(float(r.entry_dema), 1),
            "entry_dema_slope": round(float(r.entry_dema_slope), 4),
            "entry_gap_pts": round(float(r.entry_gap_pts), 1),
            "entry_atr": round(float(r.entry_atr), 2),
            "session": str(r.session),
            "is_win": bool(r.is_win),
            "bars_held": int(r.bars_held),
            "duration_min": int(r.duration_min),
        })
    return out


def summary_record(df: pd.DataFrame, start: str, end: str,
                   timeframe: str, candles: int) -> dict:
    return {
        "strategy": "FVG Scalper",
        "timeframe": timeframe,
        "period": f"{start} to {end}",
        "total_trades": len(df),
        "total_candles": candles,
        "wins": int(df["is_win"].sum()) if not df.empty else 0,
        "pnl_usd": float(df["pnl_usd"].sum()) if not df.empty else 0.0,
        "win_rate": float(df["is_win"].mean() * 100) if not df.empty else 0.0,
        "avg_r": float(df["r_multiple"].mean()) if not df.empty else 0.0,
        "profit_factor": (float(df[df["pnl_usd"] > 0]["pnl_usd"].sum()) /
                          abs(float(df[df["pnl_usd"] < 0]["pnl_usd"].sum()))
                          if not df.empty and abs(float(df[df["pnl_usd"] < 0]["pnl_usd"].sum())) > 0
                          else 0.0),
    }


def export_ui(bars_by_tf, events_by_tf, start, end):
    OUT_CANDLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    for timeframe, df_bars in bars_by_tf.items():
        candles = candle_records(df_bars, start, end)
        summary = summary_record(events_by_tf[timeframe], start, end,
                                 timeframe, len(candles))
        out_path = ROOT / f"ui/data/candles_{STRATEGY_KEY}_{timeframe}.json"
        out_path.write_text(json.dumps({"summary": summary, "candles": candles},
                                       separators=(",", ":")))
        trades_path = ROOT / f"ui/data/trade_events_{STRATEGY_KEY}_{timeframe}.json"
        trades_path.write_text(json.dumps({"summary": summary, "trades": trade_records(events_by_tf[timeframe])},
                                          separators=(",", ":")))
    OUT_TRADES_JSON.write_text(
        (ROOT / f"ui/data/trade_events_{STRATEGY_KEY}_5m.json").read_text())


# ── main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"Loading 1m OHLCV from {args.raw_db}...")
    df_1m = load_ohlcv_1m(args.raw_db, args.start, args.end, args.table)
    bars_by_tf = {
        "1m": df_1m.set_index("timestamp_utc").sort_index(),
        "5m": resample_5m(df_1m),
        "15m": resample_ohlcv(df_1m, "15min"),
    }
    print(f"Loaded {len(df_1m):,} 1m rows -> "
          f"{len(bars_by_tf['5m']):,} 5m / {len(bars_by_tf['15m']):,} 15m candles")

    events_by_tf = {}
    for tf, bars in bars_by_tf.items():
        tf_min = {"1m": 1, "5m": 5, "15m": 15}[tf]
        events = build_events(bars, args.start, tf_min)
        events_by_tf[tf] = events
        if tf == "5m":
            events.to_parquet(args.out)
            print(f"Saved {len(events)} trade events -> {args.out}")
        print(f"{tf}: trades={len(events)}, "
              f"pnl=${events['pnl_usd'].sum():.2f}" if not events.empty else f"{tf}: 0 trades",
              f", win_rate={events['is_win'].mean()*100:.1f}%" if not events.empty else "",
              f", avg_R={events['r_multiple'].mean():.3f}" if not events.empty else "")

    if args.export_ui:
        export_ui(bars_by_tf, events_by_tf, args.start, args.end)
        print(f"Saved UI candles -> ui/data/candles_{STRATEGY_KEY}_{{1m,5m,15m}}.json")
        print(f"Saved UI trades -> ui/data/trade_events_{STRATEGY_KEY}_{{1m,5m,15m}}.json")


if __name__ == "__main__":
    main()
