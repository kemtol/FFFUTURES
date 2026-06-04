#!/usr/bin/env python3
"""P0 sweep for NASDAQ ORB short-switch TP2R variants.

The baseline short-switch comparison only varied short TP. This runner keeps
short TP fixed at 2R and sweeps the first practical optimization knobs:

- short-side SuperTrend filter
- short probe risk
- switch-to-long risk
- switch trigger buffer
- short entry time guard

Long-first breakouts remain allowed at the baseline $500 risk. If a short
breakout is rejected by filter/time guard, the day is still allowed to take a
later long breakout. That keeps the natural NASDAQ long bias intact while
testing whether the short leg can add value.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from build_short_reversal_switch_comparison import (
    OUTPUT_DIR,
    build_leg,
    load_baseline_events,
    size_contracts,
)
from build_supertrend_regime_features import build_st_table, load_anchor, max_drawdown, profit_factor
from common import assert_mnq_namespaces, load_config, project_path, write_json
from sweep_orb_params import hhmm_to_minutes, load_l1, minutes_to_hhmm

MODEL_OUTPUT_DIR = "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
SUMMARY_CSV = "short_switch_tp2r_p0_sweep.csv"
SUMMARY_PARQUET = "short_switch_tp2r_p0_sweep.parquet"
BEST_EVENTS_CSV = "short_switch_tp2r_p0_best_events.csv"
BEST_LEGS_CSV = "short_switch_tp2r_p0_best_legs.csv"
BEST_YEARLY_CSV = "short_switch_tp2r_p0_best_yearly.csv"
BEST_MONTHLY_CSV = "short_switch_tp2r_p0_best_monthly.csv"
REPORT_MD = "short_switch_tp2r_p0_sweep.md"
FULL_REPORT_MD = "short_switch_tp2r_p0_full_report.md"
MANIFEST_JSON = "short_switch_tp2r_p0_sweep_manifest.json"
CHART_STEMS = [
    "short_switch_tp2r_p0_equity_curve",
    "short_switch_tp2r_p0_drawdown_curve",
    "short_switch_tp2r_p0_monthly_pnl_2026",
    "short_switch_tp2r_p0_rolling_windows",
    "short_switch_tp2r_p0_last30_equity",
]

WINDOW_DAYS = [5, 10, 20, 30, 50, 100, 200]
SHORT_TP_R = 2.0
LONG_FIRST_RISK_USD = 500.0
LONG_TP_R = 2.0
ST_FACTOR = 4.0
SHORT_FILTERS: dict[str, list[tuple[int, int]]] = {
    "none": [],
    "st5_50_bearish": [(5, 50)],
    "st5_20_bearish": [(5, 20)],
    "st5_50_and_st15_20_bearish": [(5, 50), (15, 20)],
}
SHORT_RISKS = [250.0, 350.0, 500.0]
SWITCH_LONG_RISKS = [500.0, 750.0]
SWITCH_BUFFERS = ["0", "2ticks", "0.25r"]
SHORT_ENTRY_UNTIL = ["none", "10:30", "11:00"]
P0_CHART_COLORS = {
    "long_only_no_st": "#334155",
    "p0_none_sr500_lr500_buf0_tgnone": "#7c3aed",
    "p0_st5_20_bearish_sr350_lr750_buf0_tg1030": "#0f766e",
}


@dataclass(frozen=True)
class VariantSpec:
    short_filter: str
    short_risk_usd: float
    switch_long_risk_usd: float
    switch_buffer_mode: str
    short_entry_until: str

    @property
    def variant_id(self) -> str:
        guard = self.short_entry_until.replace(":", "")
        return (
            f"p0_{self.short_filter}"
            f"_sr{int(self.short_risk_usd)}"
            f"_lr{int(self.switch_long_risk_usd)}"
            f"_buf{self.switch_buffer_mode.replace('.', '_')}"
            f"_tg{guard}"
        )

    @property
    def label(self) -> str:
        return (
            f"P0 {self.short_filter}, short ${self.short_risk_usd:g}, "
            f"switch long ${self.switch_long_risk_usd:g}, "
            f"buffer {self.switch_buffer_mode}, guard {self.short_entry_until}"
        )


class SuperTrendLookup:
    def __init__(self, l1: pd.DataFrame, requirements: list[tuple[int, int]]) -> None:
        self.tables: dict[tuple[int, int], dict[str, Any]] = {}
        if not requirements:
            return
        context = l1[
            [
                "timestamp_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "bar_data_quality_ok",
            ]
        ].copy()
        context["timestamp_utc"] = pd.to_datetime(context["timestamp_utc"], utc=True)
        context = context.sort_values("timestamp_utc").reset_index(drop=True)

        periods_by_timeframe: dict[int, list[int]] = {}
        for timeframe, period in requirements:
            periods_by_timeframe.setdefault(timeframe, []).append(period)

        for timeframe, periods in periods_by_timeframe.items():
            periods = sorted(set(periods))
            table, _ = build_st_table(context, timeframe, periods, factor=ST_FACTOR)
            ts_col = f"st{timeframe}_feature_ts"
            for period in periods:
                dir_col = f"st{timeframe}_{period}_dir"
                clean = table[[ts_col, dir_col]].dropna(subset=[ts_col]).copy()
                clean[ts_col] = pd.to_datetime(clean[ts_col], utc=True)
                self.tables[(timeframe, period)] = {
                    "times": clean[ts_col].astype("int64").to_numpy(),
                    "timestamps": clean[ts_col].to_numpy(),
                    "directions": clean[dir_col].astype("int16").to_numpy(),
                }

    def lookup_dir(self, signal_ts: pd.Timestamp, timeframe: int, period: int) -> tuple[int | None, pd.Timestamp | None, float | None]:
        table = self.tables[(timeframe, period)]
        ts_ns = pd.Timestamp(signal_ts).value
        idx = table["times"].searchsorted(ts_ns, side="right") - 1
        if idx < 0:
            return None, None, None
        feature_ts = pd.Timestamp(table["timestamps"][idx])
        if feature_ts.tzinfo is None:
            feature_ts = feature_ts.tz_localize("UTC")
        else:
            feature_ts = feature_ts.tz_convert("UTC")
        lag_minutes = (pd.Timestamp(signal_ts) - feature_ts).total_seconds() / 60.0
        return int(table["directions"][idx]), feature_ts, float(lag_minutes)

    def short_filter_pass(self, filter_id: str, signal_ts: pd.Timestamp) -> tuple[bool, float | None, int]:
        requirements = SHORT_FILTERS[filter_id]
        if not requirements:
            return True, None, 0
        max_lag: float | None = None
        lookahead_violations = 0
        for timeframe, period in requirements:
            direction, _, lag_minutes = self.lookup_dir(signal_ts, timeframe, period)
            if lag_minutes is not None and lag_minutes < 0:
                lookahead_violations += 1
            if direction is None or direction != 1:
                return False, max_lag, lookahead_violations
            if lag_minutes is not None:
                max_lag = lag_minutes if max_lag is None else max(max_lag, lag_minutes)
        return True, max_lag, lookahead_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--orb-minutes", type=int, default=15)
    parser.add_argument("--short-filters", default=",".join(SHORT_FILTERS))
    parser.add_argument("--short-risks", default=",".join(str(int(x)) for x in SHORT_RISKS))
    parser.add_argument("--switch-long-risks", default=",".join(str(int(x)) for x in SWITCH_LONG_RISKS))
    parser.add_argument("--switch-buffers", default=",".join(SWITCH_BUFFERS))
    parser.add_argument("--short-entry-until", default=",".join(SHORT_ENTRY_UNTIL))
    return parser.parse_args()


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def switch_buffer_pts(mode: str, entry_risk_pts: float, cfg: dict) -> float:
    if mode == "0":
        return 0.0
    if mode == "2ticks":
        return 2.0 * float(cfg["costs"]["tick_size"])
    if mode == "0.25r":
        return 0.25 * entry_risk_pts
    raise ValueError(f"Unsupported switch buffer mode: {mode}")


def scan_short_exit_with_buffer(
    *,
    day: pd.DataFrame,
    signal_bar: pd.Series,
    entry_idx: int,
    entry_price: float,
    entry_risk_pts: float,
    orb_high: float,
    time_exit: str,
    buffer_pts: float,
) -> tuple[pd.Series, str, float, int | None, pd.Series | None]:
    exit_scan = day.iloc[entry_idx:]
    exit_scan = exit_scan[exit_scan["bar_data_quality_ok"].astype(bool)]
    time_exit_candidates = exit_scan[exit_scan["ny_time"] >= time_exit]
    if time_exit_candidates.empty:
        raise ValueError("missing time exit")
    time_exit_idx = int(time_exit_candidates.iloc[0].name)
    tp_price = entry_price - SHORT_TP_R * entry_risk_pts
    switch_threshold = orb_high + buffer_pts

    for idx, row in exit_scan.iterrows():
        if idx > time_exit_idx:
            break
        if float(row["low"]) <= tp_price:
            return row, "TP_2R", tp_price, None, None
        if idx == time_exit_idx or row["ny_time"] >= time_exit:
            return row, "TIME_EXIT", float(row["close"]), None, None
        if float(row["close"]) > switch_threshold:
            switch_entry_idx = int(idx) + 1
            if switch_entry_idx < len(day):
                switch_entry_bar = day.iloc[switch_entry_idx]
                if bool(switch_entry_bar["bar_data_quality_ok"]) and switch_entry_bar["ny_time"] <= time_exit:
                    return switch_entry_bar, "SWITCH_TO_LONG", float(switch_entry_bar["open"]), switch_entry_idx, row
    time_exit_bar = time_exit_candidates.iloc[0]
    return time_exit_bar, "TIME_EXIT", float(time_exit_bar["close"]), None, None


def scan_long_exit_fast(
    package: dict[str, Any],
    *,
    entry_idx: int,
    entry_price: float,
    entry_risk_pts: float,
    time_exit_min: int,
) -> tuple[int, str, float]:
    quality_indices = package["quality_indices"]
    start_pos = quality_indices.searchsorted(entry_idx, side="left")
    scan = quality_indices[start_pos:]
    if len(scan) == 0:
        raise ValueError("missing exit scan")
    time_candidates = scan[package["ny_time_minutes"][scan] >= time_exit_min]
    if len(time_candidates) == 0:
        raise ValueError("missing time exit")
    time_exit_idx = int(time_candidates[0])
    scan = scan[scan <= time_exit_idx]
    tp_price = entry_price + LONG_TP_R * entry_risk_pts
    tp_hits = scan[package["high"][scan] >= tp_price]
    if len(tp_hits) and int(tp_hits[0]) <= time_exit_idx:
        return int(tp_hits[0]), "TP_2R", float(tp_price)
    return time_exit_idx, "TIME_EXIT", float(package["close"][time_exit_idx])


def scan_short_exit_with_buffer_fast(
    package: dict[str, Any],
    *,
    entry_idx: int,
    entry_price: float,
    entry_risk_pts: float,
    time_exit_min: int,
    buffer_pts: float,
) -> tuple[int, str, float, int | None, int | None]:
    quality_indices = package["quality_indices"]
    start_pos = quality_indices.searchsorted(entry_idx, side="left")
    scan = quality_indices[start_pos:]
    if len(scan) == 0:
        raise ValueError("missing exit scan")
    time_candidates = scan[package["ny_time_minutes"][scan] >= time_exit_min]
    if len(time_candidates) == 0:
        raise ValueError("missing time exit")
    time_exit_idx = int(time_candidates[0])
    scan = scan[scan <= time_exit_idx]
    tp_price = entry_price - SHORT_TP_R * entry_risk_pts
    tp_hits = scan[package["low"][scan] <= tp_price]
    tp_idx = int(tp_hits[0]) if len(tp_hits) else None

    switch_threshold = package["orb_high"] + buffer_pts
    switch_hits = scan[
        (scan < time_exit_idx)
        & (package["close"][scan] > switch_threshold)
    ]
    valid_switches: list[tuple[int, int]] = []
    for switch_signal_idx in switch_hits:
        switch_entry_idx = int(switch_signal_idx) + 1
        if switch_entry_idx < len(package["day"]):
            if bool(package["quality"][switch_entry_idx]) and int(package["ny_time_minutes"][switch_entry_idx]) <= time_exit_min:
                valid_switches.append((int(switch_signal_idx), switch_entry_idx))
                break
    switch_idx, switch_entry_idx = valid_switches[0] if valid_switches else (None, None)

    if tp_idx is not None and (switch_idx is None or tp_idx <= switch_idx) and tp_idx <= time_exit_idx:
        return tp_idx, "TP_2R", float(tp_price), None, None
    if switch_idx is not None and switch_entry_idx is not None:
        return (
            switch_entry_idx,
            "SWITCH_TO_LONG",
            float(package["open"][switch_entry_idx]),
            switch_entry_idx,
            switch_idx,
        )
    return time_exit_idx, "TIME_EXIT", float(package["close"][time_exit_idx]), None, None


def build_day_packages(l1: pd.DataFrame, cfg: dict, orb_minutes: int) -> list[dict[str, Any]]:
    rules = cfg["rules"]
    session = cfg["session"]
    market_open_min = hhmm_to_minutes(session["market_open"])
    time_exit_min = hhmm_to_minutes(session["time_exit"])
    min_orb_range = float(rules["min_orb_range_pts"])
    max_orb_range = float(rules["max_orb_range_pts"])
    post_end_offset = time_exit_min - market_open_min

    packages = []
    for ny_date, day in l1.groupby("ny_date", sort=True):
        day = day.reset_index(drop=True)
        quality_mask = day["bar_data_quality_ok"].astype(bool).to_numpy()
        minutes_from_open = day["minutes_from_open"].astype(int).to_numpy()
        close = day["close"].astype(float).to_numpy()
        high = day["high"].astype(float).to_numpy()
        low = day["low"].astype(float).to_numpy()
        open_ = day["open"].astype(float).to_numpy()
        ny_time_minutes = day["ny_time"].map(hhmm_to_minutes).astype(int).to_numpy()
        timestamps = pd.to_datetime(day["timestamp_utc"], utc=True)

        orb_mask = quality_mask & (minutes_from_open > 0) & (minutes_from_open <= orb_minutes)
        orb = day.loc[orb_mask]
        if len(orb) != orb_minutes:
            continue
        orb_high = float(orb["high"].max())
        orb_low = float(orb["low"].min())
        orb_range = orb_high - orb_low
        if orb_range < min_orb_range or orb_range > max_orb_range:
            continue

        post_indices = np.flatnonzero(
            quality_mask & (minutes_from_open > orb_minutes) & (minutes_from_open < post_end_offset)
        )
        if len(post_indices) == 0:
            continue
        candidate_mask = (close[post_indices] > orb_high) | (close[post_indices] < orb_low)
        candidate_indices = post_indices[candidate_mask]
        if len(candidate_indices) == 0:
            continue
        packages.append(
            {
                "ny_date": ny_date,
                "day": day,
                "quality": quality_mask,
                "quality_indices": np.flatnonzero(quality_mask),
                "minutes_from_open": minutes_from_open,
                "ny_time_minutes": ny_time_minutes,
                "timestamp_utc": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "candidate_indices": candidate_indices,
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_range": orb_range,
            }
        )
    return packages


def select_first_accepted_signal(
    package: dict[str, Any],
    spec: VariantSpec,
    st_lookup: SuperTrendLookup,
    time_exit: str,
) -> tuple[str, int, pd.Series, dict[str, Any]] | None:
    orb_high = package["orb_high"]
    orb_low = package["orb_low"]
    close = package["close"]
    ny_time_minutes = package["ny_time_minutes"]
    timestamps = package["timestamp_utc"]
    guard_min = hhmm_to_minutes(spec.short_entry_until) if spec.short_entry_until != "none" else None
    stats = {
        "short_filter_evaluations": 0,
        "short_filter_rejections": 0,
        "short_time_guard_rejections": 0,
        "short_filter_lookahead_violations": 0,
        "short_filter_max_lag_minutes": None,
    }

    for idx in package["candidate_indices"]:
        cur_close = float(close[idx])
        row = package["day"].iloc[int(idx)]
        if cur_close > orb_high:
            return "LONG", int(idx), row, stats
        if cur_close < orb_low:
            if guard_min is not None and int(ny_time_minutes[idx]) > guard_min:
                stats["short_time_guard_rejections"] += 1
                continue
            stats["short_filter_evaluations"] += 1
            signal_ts = pd.Timestamp(timestamps.iloc[int(idx)])
            passed, max_lag, violations = st_lookup.short_filter_pass(spec.short_filter, signal_ts)
            stats["short_filter_lookahead_violations"] += violations
            if max_lag is not None:
                current = stats["short_filter_max_lag_minutes"]
                stats["short_filter_max_lag_minutes"] = max_lag if current is None else max(current, max_lag)
            if not passed:
                stats["short_filter_rejections"] += 1
                continue
            return "SHORT", int(idx), row, stats
    return None


def simulate_variant(
    packages: list[dict[str, Any]],
    cfg: dict,
    st_lookup: SuperTrendLookup,
    spec: VariantSpec,
    orb_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rules = cfg["rules"]
    costs = cfg["costs"]
    session = cfg["session"]
    time_exit = session["time_exit"]
    time_exit_min = hhmm_to_minutes(time_exit)
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    min_entry_risk = float(rules["min_entry_risk_pts"])
    max_entry_risk = float(rules["max_entry_risk_pts"])

    sequences = []
    legs = []
    aggregate_stats = {
        "short_filter_evaluations": 0,
        "short_filter_rejections": 0,
        "short_time_guard_rejections": 0,
        "short_filter_lookahead_violations": 0,
        "short_filter_max_lag_minutes": None,
    }

    for package in packages:
        day = package["day"]
        selected = select_first_accepted_signal(package, spec, st_lookup, time_exit)
        if selected is None:
            continue
        first_side, signal_idx, signal_bar, stats = selected
        for key in [
            "short_filter_evaluations",
            "short_filter_rejections",
            "short_time_guard_rejections",
            "short_filter_lookahead_violations",
        ]:
            aggregate_stats[key] += int(stats[key])
        if stats["short_filter_max_lag_minutes"] is not None:
            current = aggregate_stats["short_filter_max_lag_minutes"]
            aggregate_stats["short_filter_max_lag_minutes"] = (
                stats["short_filter_max_lag_minutes"]
                if current is None
                else max(current, stats["short_filter_max_lag_minutes"])
            )

        entry_idx = signal_idx + 1
        if entry_idx >= len(day):
            continue
        entry_bar = day.iloc[entry_idx]
        if (not bool(entry_bar["bar_data_quality_ok"])) or entry_bar["ny_time"] > time_exit:
            continue

        orb_high = package["orb_high"]
        orb_low = package["orb_low"]
        orb_range = package["orb_range"]
        sequence_id = (
            f"NASDAQ_ORB_{orb_minutes}m_p0_short_switch_tp2r_"
            f"{spec.variant_id}_{package['ny_date']}"
        )
        leg_rows: list[dict[str, Any]] = []
        try:
            if first_side == "LONG":
                entry_price = float(entry_bar["open"]) + slippage_pts
                stop_reference = orb_low
                entry_risk_pts = entry_price - stop_reference
                if entry_risk_pts < min_entry_risk or entry_risk_pts > max_entry_risk:
                    continue
                contracts, risk_per_contract, contracts_float = size_contracts(
                    entry_risk_pts, cfg, LONG_FIRST_RISK_USD
                )
                if contracts <= 0:
                    continue
                exit_idx, exit_reason, raw_exit_price = scan_long_exit_fast(
                    package,
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    entry_risk_pts=entry_risk_pts,
                    time_exit_min=time_exit_min,
                )
                exit_bar = day.iloc[exit_idx]
                leg_rows.append(
                    build_leg(
                        variant_id=spec.variant_id,
                        ny_date=package["ny_date"],
                        sequence_id=sequence_id,
                        leg_index=1,
                        side="LONG",
                        signal_bar=signal_bar,
                        entry_bar=entry_bar,
                        exit_bar=exit_bar,
                        exit_reason=exit_reason,
                        raw_exit_price=raw_exit_price,
                        entry_price=entry_price,
                        stop_reference=stop_reference,
                        entry_risk_pts=entry_risk_pts,
                        contracts=contracts,
                        risk_per_contract=risk_per_contract,
                        contracts_float=contracts_float,
                        cfg=cfg,
                        target_risk=LONG_FIRST_RISK_USD,
                        orb_high=orb_high,
                        orb_low=orb_low,
                        orb_range=orb_range,
                        orb_minutes=orb_minutes,
                        short_tp_r=SHORT_TP_R,
                    )
                )
            else:
                entry_price = float(entry_bar["open"]) - slippage_pts
                stop_reference = orb_high
                entry_risk_pts = stop_reference - entry_price
                if entry_risk_pts < min_entry_risk or entry_risk_pts > max_entry_risk:
                    continue
                contracts, risk_per_contract, contracts_float = size_contracts(
                    entry_risk_pts, cfg, spec.short_risk_usd
                )
                if contracts <= 0:
                    continue
                buffer_pts = switch_buffer_pts(spec.switch_buffer_mode, entry_risk_pts, cfg)
                exit_idx, exit_reason, raw_exit_price, switch_entry_idx, switch_signal_idx = (
                    scan_short_exit_with_buffer_fast(
                        package,
                        entry_idx=entry_idx,
                        entry_price=entry_price,
                        entry_risk_pts=entry_risk_pts,
                        time_exit_min=time_exit_min,
                        buffer_pts=buffer_pts,
                    )
                )
                exit_bar = day.iloc[exit_idx]
                leg_rows.append(
                    build_leg(
                        variant_id=spec.variant_id,
                        ny_date=package["ny_date"],
                        sequence_id=sequence_id,
                        leg_index=1,
                        side="SHORT",
                        signal_bar=signal_bar,
                        entry_bar=entry_bar,
                        exit_bar=exit_bar,
                        exit_reason=exit_reason,
                        raw_exit_price=raw_exit_price,
                        entry_price=entry_price,
                        stop_reference=stop_reference,
                        entry_risk_pts=entry_risk_pts,
                        contracts=contracts,
                        risk_per_contract=risk_per_contract,
                        contracts_float=contracts_float,
                        cfg=cfg,
                        target_risk=spec.short_risk_usd,
                        orb_high=orb_high,
                        orb_low=orb_low,
                        orb_range=orb_range,
                        orb_minutes=orb_minutes,
                        short_tp_r=SHORT_TP_R,
                    )
                )
                if switch_entry_idx is not None and switch_signal_idx is not None:
                    long_entry_bar = day.iloc[switch_entry_idx]
                    switch_signal_bar = day.iloc[switch_signal_idx]
                    long_entry_price = float(long_entry_bar["open"]) + slippage_pts
                    long_stop_reference = orb_low
                    long_entry_risk_pts = long_entry_price - long_stop_reference
                    if min_entry_risk <= long_entry_risk_pts <= max_entry_risk:
                        long_contracts, long_risk_per_contract, long_contracts_float = size_contracts(
                            long_entry_risk_pts, cfg, spec.switch_long_risk_usd
                        )
                        if long_contracts > 0:
                            long_exit_idx, long_exit_reason, long_raw_exit_price = scan_long_exit_fast(
                                package,
                                entry_idx=switch_entry_idx,
                                entry_price=long_entry_price,
                                entry_risk_pts=long_entry_risk_pts,
                                time_exit_min=time_exit_min,
                            )
                            long_exit_bar = day.iloc[long_exit_idx]
                            leg_rows.append(
                                build_leg(
                                    variant_id=spec.variant_id,
                                    ny_date=package["ny_date"],
                                    sequence_id=sequence_id,
                                    leg_index=2,
                                    side="LONG",
                                    signal_bar=switch_signal_bar,
                                    entry_bar=long_entry_bar,
                                    exit_bar=long_exit_bar,
                                    exit_reason=long_exit_reason,
                                    raw_exit_price=long_raw_exit_price,
                                    entry_price=long_entry_price,
                                    stop_reference=long_stop_reference,
                                    entry_risk_pts=long_entry_risk_pts,
                                    contracts=long_contracts,
                                    risk_per_contract=long_risk_per_contract,
                                    contracts_float=long_contracts_float,
                                    cfg=cfg,
                                    target_risk=spec.switch_long_risk_usd,
                                    orb_high=orb_high,
                                    orb_low=orb_low,
                                    orb_range=orb_range,
                                    orb_minutes=orb_minutes,
                                    short_tp_r=SHORT_TP_R,
                                )
                            )
        except ValueError:
            continue

        if not leg_rows:
            continue
        legs.extend(leg_rows)
        pnl = sum(float(x["pnl_usd"]) for x in leg_rows)
        short_pnl = sum(float(x["pnl_usd"]) for x in leg_rows if x["side"] == "SHORT")
        long_pnl = sum(float(x["pnl_usd"]) for x in leg_rows if x["side"] == "LONG")
        switch_long_pnl = sum(
            float(x["pnl_usd"]) for x in leg_rows if x["side"] == "LONG" and int(x["leg_index"]) == 2
        )
        sequences.append(
            {
                "variant_id": spec.variant_id,
                "variant_label": spec.label,
                "sequence_id": sequence_id,
                "ny_date": package["ny_date"],
                "orb_minutes": orb_minutes,
                "short_tp_r": SHORT_TP_R,
                "long_tp_r": LONG_TP_R,
                "short_filter": spec.short_filter,
                "short_risk_usd": spec.short_risk_usd,
                "switch_long_risk_usd": spec.switch_long_risk_usd,
                "switch_buffer_mode": spec.switch_buffer_mode,
                "short_entry_until": spec.short_entry_until,
                "first_side": first_side,
                "switched_to_long": bool(any(x["exit_reason"] == "SWITCH_TO_LONG" for x in leg_rows)),
                "leg_count": len(leg_rows),
                "signal_ts": leg_rows[0]["signal_ts"],
                "entry_ts": leg_rows[0]["entry_ts"],
                "exit_ts": leg_rows[-1]["exit_ts"],
                "exit_reason": "+".join(x["exit_reason"] for x in leg_rows),
                "side": first_side,
                "long_pnl_usd": long_pnl,
                "short_pnl_usd": short_pnl,
                "switch_long_pnl_usd": switch_long_pnl,
                "pnl_usd": pnl,
                "pnl_net_usd": pnl,
                "label": int(pnl > 0),
                "contracts_used": sum(int(x["contracts_used"]) for x in leg_rows),
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_range_pts": orb_range,
            }
        )

    seq_df = pd.DataFrame(sequences)
    leg_df = pd.DataFrame(legs)
    if not seq_df.empty:
        seq_df = seq_df.sort_values("signal_ts").reset_index(drop=True)
    if not leg_df.empty:
        leg_df = leg_df.sort_values(["signal_ts", "leg_index"]).reset_index(drop=True)
    return seq_df, leg_df, aggregate_stats


def subset_dates(events: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(events["ny_date"])
    return events[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def summarize_events(
    events: pd.DataFrame,
    legs: pd.DataFrame,
    anchor: pd.Timestamp,
    spec: VariantSpec | None,
    *,
    label: str,
    variant_id: str,
    extra_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pnl = events["pnl_net_usd"].astype(float) if not events.empty else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total = float(pnl.sum()) if not pnl.empty else 0.0
    dd = max_drawdown(pnl)
    row: dict[str, Any] = {
        "variant_id": variant_id,
        "label": label,
        "short_filter": spec.short_filter if spec else "baseline",
        "short_risk_usd": spec.short_risk_usd if spec else 0.0,
        "switch_long_risk_usd": spec.switch_long_risk_usd if spec else 0.0,
        "switch_buffer_mode": spec.switch_buffer_mode if spec else "baseline",
        "short_entry_until": spec.short_entry_until if spec else "baseline",
        "trades": int(len(events)),
        "win_rate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
        "pnl_usd": total,
        "max_dd_usd": dd,
        "return_dd": total / abs(dd) if dd else None,
        "profit_factor": profit_factor(pnl),
        "avg_trade_usd": float(pnl.mean()) if not pnl.empty else 0.0,
        "long_first_trades": int(events["first_side"].eq("LONG").sum()) if "first_side" in events else 0,
        "short_first_trades": int(events["first_side"].eq("SHORT").sum()) if "first_side" in events else 0,
        "switch_count": int(events["switched_to_long"].fillna(False).sum()) if "switched_to_long" in events else 0,
        "long_pnl_usd": float(events.get("long_pnl_usd", pd.Series(0.0, index=events.index)).sum()),
        "short_pnl_usd": float(events.get("short_pnl_usd", pd.Series(0.0, index=events.index)).sum()),
        "switch_long_pnl_usd": float(events.get("switch_long_pnl_usd", pd.Series(0.0, index=events.index)).sum()),
        "gross_profit_usd": float(wins.sum()) if not wins.empty else 0.0,
        "gross_loss_usd": float(losses.sum()) if not losses.empty else 0.0,
        "legs": int(len(legs)),
        "leg_short_pnl_usd": float(legs.loc[legs["side"].eq("SHORT"), "pnl_usd"].sum()) if not legs.empty else 0.0,
        "leg_switch_long_pnl_usd": float(
            legs.loc[legs["side"].eq("LONG") & legs["leg_index"].eq(2), "pnl_usd"].sum()
        )
        if not legs.empty
        else 0.0,
    }
    for prefix, subset in [
        ("jan_may_2026", subset_dates(events, "2026-01-01", "2026-05-31")),
        ("march_2026", subset_dates(events, "2026-03-01", "2026-03-31")),
    ]:
        sub_pnl = subset["pnl_net_usd"].astype(float)
        row[f"{prefix}_trades"] = int(len(subset))
        row[f"{prefix}_pnl_usd"] = float(sub_pnl.sum()) if not sub_pnl.empty else 0.0
        row[f"{prefix}_max_dd_usd"] = max_drawdown(sub_pnl) if not sub_pnl.empty else 0.0
        row[f"{prefix}_win_rate"] = float((sub_pnl > 0).mean()) if not sub_pnl.empty else 0.0
    for days in WINDOW_DAYS:
        window = events[(events["signal_ts"] > anchor - pd.Timedelta(days=days)) & (events["signal_ts"] <= anchor)]
        w_pnl = window["pnl_net_usd"].astype(float)
        row[f"last_{days}d_trades"] = int(len(window))
        row[f"last_{days}d_pnl_usd"] = float(w_pnl.sum()) if not w_pnl.empty else 0.0
        row[f"last_{days}d_max_dd_usd"] = max_drawdown(w_pnl) if not w_pnl.empty else 0.0
    if extra_stats:
        row.update(extra_stats)
    return row


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    baseline = out[out["variant_id"].eq("long_only_no_st")].iloc[0]
    for col in ["pnl_usd", "max_dd_usd", "march_2026_pnl_usd", "march_2026_max_dd_usd"]:
        out[f"{col}_delta_vs_baseline"] = out[col] - baseline[col]
    for days in WINDOW_DAYS:
        out[f"last_{days}d_pnl_delta_vs_baseline"] = out[f"last_{days}d_pnl_usd"] - baseline[f"last_{days}d_pnl_usd"]
        out[f"last_{days}d_dd_delta_vs_baseline"] = out[f"last_{days}d_max_dd_usd"] - baseline[f"last_{days}d_max_dd_usd"]
    out["p0_recent_score"] = (
        out["last_30d_pnl_usd"]
        + out["last_30d_max_dd_usd"].clip(upper=0.0)
        + 0.25 * out["march_2026_pnl_usd_delta_vs_baseline"]
        + 0.10 * out["pnl_usd_delta_vs_baseline"]
    )
    out["beats_baseline_30d_pnl"] = out["last_30d_pnl_usd"] > baseline["last_30d_pnl_usd"]
    out["beats_baseline_30d_dd"] = out["last_30d_max_dd_usd"] > baseline["last_30d_max_dd_usd"]
    out["improves_march_pnl"] = out["march_2026_pnl_usd"] > baseline["march_2026_pnl_usd"]
    return out


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${float(value):,.0f}"


def number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def raw_url(path: str) -> str:
    return (
        "https://raw.githubusercontent.com/kemtol/FFFUTURES/main/"
        f"model/MNQ/ORB/rule_based_15m_long_tp2r_eod/{path}"
    )


def clean_svg(path: Path) -> None:
    text = path.read_text()
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def save_fig(fig, chart_dir: Path, stem: str) -> list[str]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    svg = chart_dir / f"{stem}.svg"
    png = chart_dir / f"{stem}.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=150)
    clean_svg(svg)
    return [str(svg.relative_to(project_path("."))), str(png.relative_to(project_path(".")))]


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def normalize_chart_events(events: pd.DataFrame, variant_id: str, label: str) -> pd.DataFrame:
    out = events.copy()
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        if col in out:
            out[col] = pd.to_datetime(out[col], utc=True)
    out["ny_date"] = pd.to_datetime(out["ny_date"])
    out["variant_id"] = variant_id
    out["variant_label"] = label
    out["pnl_net_usd"] = out["pnl_net_usd"].astype(float)
    return out.sort_values("signal_ts").reset_index(drop=True)


def chart_drawdown(pnl: pd.Series) -> pd.Series:
    equity = pnl.astype(float).cumsum().reset_index(drop=True)
    peak = pd.concat([pd.Series([0.0]), equity], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    return equity - peak


def plot_p0_equity(frames: list[pd.DataFrame], chart_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for frame in frames:
        variant_id = frame["variant_id"].iloc[0]
        equity = frame["pnl_net_usd"].cumsum()
        ax.plot(
            frame["signal_ts"],
            equity,
            label=frame["variant_label"].iloc[0],
            color=P0_CHART_COLORS.get(variant_id),
            linewidth=1.7,
        )
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "NASDAQ ORB Enhancement 3: P0 Equity Comparison", "Cumulative PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    files = save_fig(fig, chart_dir, "short_switch_tp2r_p0_equity_curve")
    plt.close(fig)
    return files


def plot_p0_drawdown(frames: list[pd.DataFrame], chart_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for frame in frames:
        variant_id = frame["variant_id"].iloc[0]
        dd = chart_drawdown(frame["pnl_net_usd"])
        ax.plot(
            frame["signal_ts"],
            dd,
            label=frame["variant_label"].iloc[0],
            color=P0_CHART_COLORS.get(variant_id),
            linewidth=1.5,
        )
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "NASDAQ ORB Enhancement 3: P0 Drawdown Comparison", "Drawdown ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    files = save_fig(fig, chart_dir, "short_switch_tp2r_p0_drawdown_curve")
    plt.close(fig)
    return files


def plot_p0_monthly_2026(frames: list[pd.DataFrame], chart_dir: Path) -> list[str]:
    months = pd.period_range("2026-01", "2026-05", freq="M").astype(str).tolist()
    x = np.arange(len(months))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for i, frame in enumerate(frames):
        variant_id = frame["variant_id"].iloc[0]
        monthly = (
            frame.assign(month=frame["ny_date"].dt.to_period("M").astype(str))
            .groupby("month")["pnl_net_usd"]
            .sum()
            .reindex(months, fill_value=0.0)
        )
        ax.bar(
            x + (i - 1) * width,
            monthly.values,
            width=width,
            label=frame["variant_label"].iloc[0],
            color=P0_CHART_COLORS.get(variant_id),
            alpha=0.88,
        )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    style_axis(ax, "NASDAQ ORB Enhancement 3: Monthly PnL 2026", "Monthly PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    files = save_fig(fig, chart_dir, "short_switch_tp2r_p0_monthly_pnl_2026")
    plt.close(fig)
    return files


def plot_p0_rolling(summary: pd.DataFrame, chart_dir: Path, best_variant_id: str) -> list[str]:
    windows = WINDOW_DAYS
    plot_ids = ["long_only_no_st", "p0_none_sr500_lr500_buf0_tgnone", best_variant_id]
    labels = {
        "long_only_no_st": "Long only baseline",
        "p0_none_sr500_lr500_buf0_tgnone": "Existing short-switch TP2R",
        best_variant_id: "Best P0 enhancement",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    for variant_id in plot_ids:
        row = summary[summary["variant_id"].eq(variant_id)]
        if row.empty:
            continue
        row = row.iloc[0]
        pnls = [row[f"last_{days}d_pnl_usd"] for days in windows]
        dds = [row[f"last_{days}d_max_dd_usd"] for days in windows]
        axes[0].plot(windows, pnls, marker="o", label=labels[variant_id], color=P0_CHART_COLORS.get(variant_id))
        axes[1].plot(windows, dds, marker="o", label=labels[variant_id], color=P0_CHART_COLORS.get(variant_id))
    axes[0].axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    axes[1].axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(axes[0], "Rolling Window PnL", "PnL ($)")
    style_axis(axes[1], "Rolling Window Max Drawdown", "Drawdown ($)")
    axes[1].set_xlabel("Lookback days")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    files = save_fig(fig, chart_dir, "short_switch_tp2r_p0_rolling_windows")
    plt.close(fig)
    return files


def plot_p0_last30(frames: list[pd.DataFrame], anchor: pd.Timestamp, chart_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for frame in frames:
        window = frame[(frame["signal_ts"] > anchor - pd.Timedelta(days=30)) & (frame["signal_ts"] <= anchor)]
        if window.empty:
            continue
        variant_id = frame["variant_id"].iloc[0]
        equity = window["pnl_net_usd"].astype(float).cumsum()
        ax.plot(
            window["signal_ts"],
            equity,
            marker="o",
            label=frame["variant_label"].iloc[0],
            color=P0_CHART_COLORS.get(variant_id),
            linewidth=1.5,
        )
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "NASDAQ ORB Enhancement 3: Last 30D Equity", "30D cumulative PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    files = save_fig(fig, chart_dir, "short_switch_tp2r_p0_last30_equity")
    plt.close(fig)
    return files


def save_p0_charts(
    *,
    baseline: pd.DataFrame,
    existing: pd.DataFrame,
    best_events: pd.DataFrame,
    summary: pd.DataFrame,
    anchor: pd.Timestamp,
    chart_dir: Path,
    best_variant_id: str,
) -> list[str]:
    frames = [
        normalize_chart_events(baseline, "long_only_no_st", "Long only baseline"),
        normalize_chart_events(existing, "p0_none_sr500_lr500_buf0_tgnone", "Existing short-switch TP2R"),
        normalize_chart_events(best_events, best_variant_id, "Best P0 enhancement"),
    ]
    outputs = []
    outputs += plot_p0_equity(frames, chart_dir)
    outputs += plot_p0_drawdown(frames, chart_dir)
    outputs += plot_p0_monthly_2026(frames, chart_dir)
    outputs += plot_p0_rolling(summary, chart_dir, best_variant_id)
    outputs += plot_p0_last30(frames, anchor, chart_dir)
    return outputs


def render_table(df: pd.DataFrame, limit: int = 12) -> str:
    cols = [
        "variant_id",
        "trades",
        "pnl_usd",
        "max_dd_usd",
        "return_dd",
        "march_2026_pnl_usd",
        "last_30d_trades",
        "last_30d_pnl_usd",
        "last_30d_max_dd_usd",
        "short_pnl_usd",
        "switch_count",
    ]
    header = "| Variant | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD | Short PnL | Switches |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for _, row in df.head(limit).iterrows():
        rows.append(
            "| {variant} | {trades:,} | {pnl} | {dd} | {retdd} | {mar} | {d30_trades:,} | {d30_pnl} | {d30_dd} | {short_pnl} | {switches:,} |".format(
                variant=row["variant_id"],
                trades=int(row["trades"]),
                pnl=money(row["pnl_usd"]),
                dd=money(row["max_dd_usd"]),
                retdd=number(row["return_dd"]),
                mar=money(row["march_2026_pnl_usd"]),
                d30_trades=int(row["last_30d_trades"]),
                d30_pnl=money(row["last_30d_pnl_usd"]),
                d30_dd=money(row["last_30d_max_dd_usd"]),
                short_pnl=money(row["short_pnl_usd"]),
                switches=int(row["switch_count"]),
            )
        )
    return "\n".join(rows)


def summarize_period_events(events: pd.DataFrame) -> dict[str, Any]:
    pnl = events["pnl_net_usd"].astype(float) if not events.empty else pd.Series(dtype=float)
    total = float(pnl.sum()) if not pnl.empty else 0.0
    dd = max_drawdown(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    return {
        "trades": int(len(events)),
        "pnl_usd": total,
        "max_dd_usd": dd,
        "return_dd": total / abs(dd) if dd else None,
        "win_rate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
        "avg_trade_usd": float(pnl.mean()) if not pnl.empty else 0.0,
        "profit_factor": profit_factor(pnl),
        "gross_profit_usd": float(wins.sum()) if not wins.empty else 0.0,
        "gross_loss_usd": float(losses.sum()) if not losses.empty else 0.0,
        "long_pnl_usd": float(events.get("long_pnl_usd", pd.Series(0.0, index=events.index)).sum()),
        "short_pnl_usd": float(events.get("short_pnl_usd", pd.Series(0.0, index=events.index)).sum()),
        "switch_long_pnl_usd": float(events.get("switch_long_pnl_usd", pd.Series(0.0, index=events.index)).sum()),
        "switch_count": int(events.get("switched_to_long", pd.Series(False, index=events.index)).fillna(False).sum()),
    }


def build_best_period_reports(best_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if best_events.empty:
        return pd.DataFrame(), pd.DataFrame()
    events = best_events.copy()
    events["ny_date"] = pd.to_datetime(events["ny_date"])

    yearly_rows = []
    for year, group in events.groupby(events["ny_date"].dt.year, sort=True):
        row = {"year": int(year)}
        row.update(summarize_period_events(group))
        yearly_rows.append(row)

    monthly_rows = []
    for month, group in events.groupby(events["ny_date"].dt.to_period("M"), sort=True):
        row = {"month": str(month)}
        row.update(summarize_period_events(group))
        monthly_rows.append(row)

    return pd.DataFrame(yearly_rows), pd.DataFrame(monthly_rows)


def render_period_table(df: pd.DataFrame, period_col: str, limit: int | None = None) -> str:
    header = (
        f"| {period_col.title()} | Trades | PnL | DD | Ret/DD | Win Rate | Avg | "
        "Long PnL | Short PnL | Switch Long PnL | Switches |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    display = df if limit is None else df.head(limit)
    for _, row in display.iterrows():
        period_value = row[period_col]
        if period_col == "year" and not pd.isna(period_value):
            period_value = int(period_value)
        rows.append(
            "| {period} | {trades:,} | {pnl} | {dd} | {retdd} | {wr} | {avg} | {long_pnl} | {short_pnl} | {switch_long_pnl} | {switches:,} |".format(
                period=period_value,
                trades=int(row["trades"]),
                pnl=money(row["pnl_usd"]),
                dd=money(row["max_dd_usd"]),
                retdd=number(row["return_dd"]),
                wr=pct(row["win_rate"]),
                avg=money(row["avg_trade_usd"]),
                long_pnl=money(row["long_pnl_usd"]),
                short_pnl=money(row["short_pnl_usd"]),
                switch_long_pnl=money(row["switch_long_pnl_usd"]),
                switches=int(row["switch_count"]),
            )
        )
    return "\n".join(rows)


def write_full_report(
    path: Path,
    best: pd.Series,
    best_events: pd.DataFrame,
    best_legs: pd.DataFrame,
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    events = best_events.copy()
    events["ny_date"] = pd.to_datetime(events["ny_date"])
    full = summarize_period_events(events)
    ytd = summarize_period_events(events[events["ny_date"].dt.year.eq(2026)])
    monthly_2026 = monthly[monthly["month"].astype(str).str.startswith("2026")].copy()
    top_months = monthly.sort_values("pnl_usd", ascending=False).head(10)
    bottom_months = monthly.sort_values("pnl_usd", ascending=True).head(10)

    lines = [
        "# NASDAQ ORB Short-Switch TP2R P0 Full Report",
        "",
        "## Candidate",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Variant ID | `{best['variant_id']}` |",
        f"| Short filter | `{best['short_filter']}` |",
        f"| Short risk | {money(best['short_risk_usd'])} |",
        f"| Switch long risk | {money(best['switch_long_risk_usd'])} |",
        f"| Switch buffer | `{best['switch_buffer_mode']}` |",
        f"| Short entry guard | `{best['short_entry_until']}` |",
        f"| Data start | {events['ny_date'].min().date()} |",
        f"| Data end | {events['ny_date'].max().date()} |",
        f"| Events | {len(best_events):,} |",
        f"| Legs | {len(best_legs):,} |",
        "",
        "## Full History And YTD",
        "",
        "| Window | Trades | PnL | DD | Ret/DD | Win Rate | Avg | Long PnL | Short PnL | Switch Long PnL | Switches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| 2019-05-06 to 2026-05-26 | {trades:,} | {pnl} | {dd} | {retdd} | {wr} | {avg} | {long_pnl} | {short_pnl} | {switch_long_pnl} | {switches:,} |".format(
            trades=int(full["trades"]),
            pnl=money(full["pnl_usd"]),
            dd=money(full["max_dd_usd"]),
            retdd=number(full["return_dd"]),
            wr=pct(full["win_rate"]),
            avg=money(full["avg_trade_usd"]),
            long_pnl=money(full["long_pnl_usd"]),
            short_pnl=money(full["short_pnl_usd"]),
            switch_long_pnl=money(full["switch_long_pnl_usd"]),
            switches=int(full["switch_count"]),
        ),
        "| YTD 2026 | {trades:,} | {pnl} | {dd} | {retdd} | {wr} | {avg} | {long_pnl} | {short_pnl} | {switch_long_pnl} | {switches:,} |".format(
            trades=int(ytd["trades"]),
            pnl=money(ytd["pnl_usd"]),
            dd=money(ytd["max_dd_usd"]),
            retdd=number(ytd["return_dd"]),
            wr=pct(ytd["win_rate"]),
            avg=money(ytd["avg_trade_usd"]),
            long_pnl=money(ytd["long_pnl_usd"]),
            short_pnl=money(ytd["short_pnl_usd"]),
            switch_long_pnl=money(ytd["switch_long_pnl_usd"]),
            switches=int(ytd["switch_count"]),
        ),
        "",
        "## Visual Comparison",
        "",
        "Visual ini membandingkan baseline, short-switch TP2R lama, dan Best P0 enhancement.",
        "",
        "### Equity Curve",
        "",
        f"![P0 Equity Comparison]({raw_url('charts/short_switch_tp2r_p0_equity_curve.png')})",
        "",
        "### Drawdown Curve",
        "",
        f"![P0 Drawdown Comparison]({raw_url('charts/short_switch_tp2r_p0_drawdown_curve.png')})",
        "",
        "### Monthly PnL 2026",
        "",
        f"![P0 Monthly PnL 2026]({raw_url('charts/short_switch_tp2r_p0_monthly_pnl_2026.png')})",
        "",
        "### Rolling Window PnL/DD",
        "",
        f"![P0 Rolling Windows]({raw_url('charts/short_switch_tp2r_p0_rolling_windows.png')})",
        "",
        "### Last 30D Equity",
        "",
        f"![P0 Last 30D Equity]({raw_url('charts/short_switch_tp2r_p0_last30_equity.png')})",
        "",
        "## Yearly",
        "",
        render_period_table(yearly, "year"),
        "",
        "## Month To Month 2026",
        "",
        render_period_table(monthly_2026, "month"),
        "",
        "## Month To Month Full History",
        "",
        render_period_table(monthly, "month"),
        "",
        "## Best Months",
        "",
        render_period_table(top_months, "month"),
        "",
        "## Worst Months",
        "",
        render_period_table(bottom_months, "month"),
        "",
        "## Artifacts",
        "",
        "| Artifact | Path |",
        "| --- | --- |",
        f"| Yearly CSV | `{manifest['artifacts']['best_yearly_csv']}` |",
        f"| Monthly CSV | `{manifest['artifacts']['best_monthly_csv']}` |",
        f"| Events CSV | `{manifest['artifacts']['best_events_csv']}` |",
        f"| Legs CSV | `{manifest['artifacts']['best_legs_csv']}` |",
        f"| Sweep report | `{manifest['artifacts']['report']}` |",
        f"| P0 chart files | `{len(manifest['artifacts'].get('charts', []))}` files |",
        "",
        "## Notes",
        "",
        "- 2019 is partial because the available artifact starts on 2019-05-06.",
        "- Drawdown is computed within each period from period-start equity, not inherited from previous periods.",
        "- This report is research-only and does not alter live execution.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_report(path: Path, summary: pd.DataFrame, best: pd.Series, manifest: dict[str, Any]) -> None:
    baseline = summary[summary["variant_id"].eq("long_only_no_st")].iloc[0]
    base_switch = summary[
        summary["variant_id"].eq("p0_none_sr500_lr500_buf0_tgnone")
    ]
    base_switch_row = base_switch.iloc[0] if not base_switch.empty else None
    top_recent = summary[~summary["variant_id"].eq("long_only_no_st")].sort_values(
        ["last_30d_pnl_usd", "last_30d_max_dd_usd", "march_2026_pnl_usd"],
        ascending=[False, False, False],
    )
    top_retdd = summary[~summary["variant_id"].eq("long_only_no_st")].sort_values(
        ["return_dd", "last_30d_pnl_usd"],
        ascending=[False, False],
    )
    top_score = summary[~summary["variant_id"].eq("long_only_no_st")].sort_values(
        ["p0_recent_score", "last_30d_pnl_usd"],
        ascending=[False, False],
    )

    lines = [
        "# NASDAQ ORB Short-Switch TP2R P0 Sweep",
        "",
        "## Scope",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Strategy family | NASDAQ Micro Futures ORB 15m short-switch TP2R |",
        "| Long-first risk | $500 fixed |",
        "| Short TP | 2R |",
        "| Long TP after switch | 2R or 15:00 NY |",
        "| Short filters | `none`, `st5_50_bearish`, `st5_20_bearish`, `st5_50_and_st15_20_bearish` |",
        "| Short risk grid | $250, $350, $500 |",
        "| Switch long risk grid | $500, $750 |",
        "| Switch buffers | `0`, `2ticks`, `0.25r` |",
        "| Short time guards | `none`, `10:30`, `11:00` |",
        "| No-lookahead rule | ST feature timestamp is selected by as-of `<= signal_ts`; entry/switch executes next M1 open |",
        f"| Variants evaluated | {manifest['rows']['summary'] - 1:,} plus baseline |",
        "",
        "## Baseline Comparison",
        "",
        "| Variant | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| Long only, no ST | {trades:,} | {pnl} | {dd} | {retdd} | {mar} | {d30_trades:,} | {d30_pnl} | {d30_dd} |".format(
            trades=int(baseline["trades"]),
            pnl=money(baseline["pnl_usd"]),
            dd=money(baseline["max_dd_usd"]),
            retdd=number(baseline["return_dd"]),
            mar=money(baseline["march_2026_pnl_usd"]),
            d30_trades=int(baseline["last_30d_trades"]),
            d30_pnl=money(baseline["last_30d_pnl_usd"]),
            d30_dd=money(baseline["last_30d_max_dd_usd"]),
        ),
    ]
    if base_switch_row is not None:
        lines.append(
            "| Existing short-switch TP2R equivalent | {trades:,} | {pnl} | {dd} | {retdd} | {mar} | {d30_trades:,} | {d30_pnl} | {d30_dd} |".format(
                trades=int(base_switch_row["trades"]),
                pnl=money(base_switch_row["pnl_usd"]),
                dd=money(base_switch_row["max_dd_usd"]),
                retdd=number(base_switch_row["return_dd"]),
                mar=money(base_switch_row["march_2026_pnl_usd"]),
                d30_trades=int(base_switch_row["last_30d_trades"]),
                d30_pnl=money(base_switch_row["last_30d_pnl_usd"]),
                d30_dd=money(base_switch_row["last_30d_max_dd_usd"]),
            )
        )
    lines.extend(
        [
            "| Best P0 score | {trades:,} | {pnl} | {dd} | {retdd} | {mar} | {d30_trades:,} | {d30_pnl} | {d30_dd} |".format(
                trades=int(best["trades"]),
                pnl=money(best["pnl_usd"]),
                dd=money(best["max_dd_usd"]),
                retdd=number(best["return_dd"]),
                mar=money(best["march_2026_pnl_usd"]),
                d30_trades=int(best["last_30d_trades"]),
                d30_pnl=money(best["last_30d_pnl_usd"]),
                d30_dd=money(best["last_30d_max_dd_usd"]),
            ),
            "",
            "## Best P0 Candidate",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Variant ID | `{best['variant_id']}` |",
            f"| Short filter | `{best['short_filter']}` |",
            f"| Short risk | {money(best['short_risk_usd'])} |",
            f"| Switch long risk | {money(best['switch_long_risk_usd'])} |",
            f"| Switch buffer | `{best['switch_buffer_mode']}` |",
            f"| Short entry guard | `{best['short_entry_until']}` |",
            f"| 30D PnL delta vs baseline | {money(best['last_30d_pnl_delta_vs_baseline'])} |",
            f"| 30D DD delta vs baseline | {money(best['last_30d_dd_delta_vs_baseline'])} |",
            f"| March PnL delta vs baseline | {money(best['march_2026_pnl_usd_delta_vs_baseline'])} |",
            f"| Beats baseline 30D PnL | {bool(best['beats_baseline_30d_pnl'])} |",
            f"| Beats baseline 30D DD | {bool(best['beats_baseline_30d_dd'])} |",
            f"| Improves March PnL | {bool(best['improves_march_pnl'])} |",
            "",
            "## Visual Comparison",
            "",
            "Visual ini menunjukkan enhancement chain: baseline control, short-switch TP2R lama, lalu kandidat P0 terbaik.",
            "",
            "### Equity Curve",
            "",
            f"![P0 Equity Comparison]({raw_url('charts/short_switch_tp2r_p0_equity_curve.png')})",
            "",
            "### Drawdown Curve",
            "",
            f"![P0 Drawdown Comparison]({raw_url('charts/short_switch_tp2r_p0_drawdown_curve.png')})",
            "",
            "### Monthly PnL 2026",
            "",
            f"![P0 Monthly PnL 2026]({raw_url('charts/short_switch_tp2r_p0_monthly_pnl_2026.png')})",
            "",
            "### Rolling Window PnL/DD",
            "",
            f"![P0 Rolling Windows]({raw_url('charts/short_switch_tp2r_p0_rolling_windows.png')})",
            "",
            "### Last 30D Equity",
            "",
            f"![P0 Last 30D Equity]({raw_url('charts/short_switch_tp2r_p0_last30_equity.png')})",
            "",
            "## Top By Recent 30D PnL",
            "",
            render_table(top_recent),
            "",
            "## Top By Full Ret/DD",
            "",
            render_table(top_retdd),
            "",
            "## Top By P0 Score",
            "",
            render_table(top_score),
            "",
            "## Audit",
            "",
            "| Check | Value |",
            "| --- | ---: |",
            f"| Total lookahead violations | {int(summary['short_filter_lookahead_violations'].fillna(0).sum()):,} |",
            f"| Max short-filter feature lag minutes | {number(summary['short_filter_max_lag_minutes'].max())} |",
            "",
            "## Artifacts",
            "",
            "| Artifact | Path |",
            "| --- | --- |",
            f"| Summary CSV | `{manifest['artifacts']['summary_csv']}` |",
            f"| Summary parquet | `{manifest['artifacts']['summary_parquet']}` |",
            f"| Best events CSV | `{manifest['artifacts']['best_events_csv']}` |",
            f"| Best legs CSV | `{manifest['artifacts']['best_legs_csv']}` |",
            f"| Best yearly CSV | `{manifest['artifacts']['best_yearly_csv']}` |",
            f"| Best monthly CSV | `{manifest['artifacts']['best_monthly_csv']}` |",
            f"| Full report | `{manifest['artifacts']['full_report']}` |",
            f"| P0 chart files | `{len(manifest['artifacts'].get('charts', []))}` files |",
            f"| Manifest | `{manifest['artifacts']['manifest']}` |",
            "",
            "## Current Read",
            "",
            "- P0 is considered better only if it improves the 30D/Topstep window while not worsening drawdown materially.",
            "- Full-history PnL alone is not enough because the current objective is evaluation-window behavior.",
            "- This is still research-only and does not change the live pipeline.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg = load_config()
    assert_mnq_namespaces(cfg)

    short_filter_ids = parse_str_list(args.short_filters)
    bad_filters = [x for x in short_filter_ids if x not in SHORT_FILTERS]
    if bad_filters:
        raise SystemExit(f"Unknown short filters: {bad_filters}")
    short_risks = parse_float_list(args.short_risks)
    switch_long_risks = parse_float_list(args.switch_long_risks)
    switch_buffers = parse_str_list(args.switch_buffers)
    short_entry_until = parse_str_list(args.short_entry_until)

    model_dir = project_path(MODEL_OUTPUT_DIR)
    data_dir = project_path(OUTPUT_DIR)
    summary_csv = model_dir / SUMMARY_CSV
    summary_parquet = data_dir / SUMMARY_PARQUET
    best_events_csv = model_dir / BEST_EVENTS_CSV
    best_legs_csv = model_dir / BEST_LEGS_CSV
    best_yearly_csv = model_dir / BEST_YEARLY_CSV
    best_monthly_csv = model_dir / BEST_MONTHLY_CSV
    chart_dir = model_dir / "charts"
    report_md = model_dir / REPORT_MD
    full_report_md = model_dir / FULL_REPORT_MD
    manifest_path = data_dir / MANIFEST_JSON
    for path in [
        summary_csv,
        summary_parquet,
        best_events_csv,
        best_legs_csv,
        best_yearly_csv,
        best_monthly_csv,
        report_md,
        full_report_md,
        manifest_path,
    ]:
        if path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing artifact without --force: {path}")

    print("loading L1 context...", flush=True)
    l1 = load_l1(cfg)
    print(f"loaded L1 rows={len(l1):,}", flush=True)
    print("building day packages...", flush=True)
    packages = build_day_packages(l1, cfg, args.orb_minutes)
    print(f"built day packages={len(packages):,}", flush=True)
    required_st = sorted(
        {
            requirement
            for filter_id in short_filter_ids
            for requirement in SHORT_FILTERS[filter_id]
        }
    )
    print(f"building ST lookup requirements={required_st}...", flush=True)
    st_lookup = SuperTrendLookup(l1, requirements=required_st)
    print("built ST lookup", flush=True)
    baseline = load_baseline_events()
    anchor = load_anchor(baseline)
    baseline_summary = summarize_events(
        baseline,
        pd.DataFrame(),
        anchor,
        None,
        label="Long only, no ST",
        variant_id="long_only_no_st",
        extra_stats={
            "short_filter_evaluations": 0,
            "short_filter_rejections": 0,
            "short_time_guard_rejections": 0,
            "short_filter_lookahead_violations": 0,
            "short_filter_max_lag_minutes": None,
        },
    )

    specs = [
        VariantSpec(short_filter, short_risk, switch_long_risk, switch_buffer, guard)
        for short_filter in short_filter_ids
        for short_risk in short_risks
        for switch_long_risk in switch_long_risks
        for switch_buffer in switch_buffers
        for guard in short_entry_until
    ]

    rows = [baseline_summary]
    for i, spec in enumerate(specs, start=1):
        events, legs, stats = simulate_variant(packages, cfg, st_lookup, spec, args.orb_minutes)
        rows.append(
            summarize_events(
                events,
                legs,
                anchor,
                spec,
                label=spec.label,
                variant_id=spec.variant_id,
                extra_stats=stats,
            )
        )
        if i % 25 == 0:
            print(f"simulated {i}/{len(specs)} variants", flush=True)

    summary = add_baseline_deltas(pd.DataFrame(rows))
    variant_summary = summary[~summary["variant_id"].eq("long_only_no_st")].copy()
    best = variant_summary.sort_values(
        ["p0_recent_score", "last_30d_pnl_usd", "last_30d_max_dd_usd"],
        ascending=[False, False, False],
    ).iloc[0]
    best_spec = VariantSpec(
        str(best["short_filter"]),
        float(best["short_risk_usd"]),
        float(best["switch_long_risk_usd"]),
        str(best["switch_buffer_mode"]),
        str(best["short_entry_until"]),
    )
    best_events, best_legs, _ = simulate_variant(packages, cfg, st_lookup, best_spec, args.orb_minutes)
    existing_spec = VariantSpec("none", 500.0, 500.0, "0", "none")
    existing_events, _, _ = simulate_variant(packages, cfg, st_lookup, existing_spec, args.orb_minutes)
    best_yearly, best_monthly = build_best_period_reports(best_events)

    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    summary.to_parquet(summary_parquet, index=False)
    best_events.to_csv(best_events_csv, index=False)
    best_legs.to_csv(best_legs_csv, index=False)
    best_yearly.to_csv(best_yearly_csv, index=False)
    best_monthly.to_csv(best_monthly_csv, index=False)
    chart_outputs = save_p0_charts(
        baseline=baseline,
        existing=existing_events,
        best_events=best_events,
        summary=summary,
        anchor=anchor,
        chart_dir=chart_dir,
        best_variant_id=str(best["variant_id"]),
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchor_ts": anchor.isoformat(),
        "params": {
            "orb_minutes": args.orb_minutes,
            "short_tp_r": SHORT_TP_R,
            "long_tp_r": LONG_TP_R,
            "long_first_risk_usd": LONG_FIRST_RISK_USD,
            "short_filters": short_filter_ids,
            "short_risks": short_risks,
            "switch_long_risks": switch_long_risks,
            "switch_buffers": switch_buffers,
            "short_entry_until": short_entry_until,
            "supertrend_factor": ST_FACTOR,
            "supertrend_requirements": {
                key: [f"ST{tf}_{period}" for tf, period in value]
                for key, value in SHORT_FILTERS.items()
            },
        },
        "rows": {
            "day_packages": int(len(packages)),
            "summary": int(len(summary)),
            "best_events": int(len(best_events)),
            "best_legs": int(len(best_legs)),
            "best_yearly": int(len(best_yearly)),
            "best_monthly": int(len(best_monthly)),
        },
        "best_variant": best.to_dict(),
        "artifacts": {
            "summary_csv": str(summary_csv.relative_to(project_path("."))),
            "summary_parquet": str(summary_parquet.relative_to(project_path("."))),
            "best_events_csv": str(best_events_csv.relative_to(project_path("."))),
            "best_legs_csv": str(best_legs_csv.relative_to(project_path("."))),
            "best_yearly_csv": str(best_yearly_csv.relative_to(project_path("."))),
            "best_monthly_csv": str(best_monthly_csv.relative_to(project_path("."))),
            "charts": chart_outputs,
            "report": str(report_md.relative_to(project_path("."))),
            "full_report": str(full_report_md.relative_to(project_path("."))),
            "manifest": str(manifest_path.relative_to(project_path("."))),
        },
    }
    write_json(manifest_path, manifest)
    write_report(report_md, summary, best, manifest)
    write_full_report(full_report_md, best, best_events, best_legs, best_yearly, best_monthly, manifest)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_parquet}")
    print(f"Wrote {best_events_csv}")
    print(f"Wrote {best_legs_csv}")
    print(f"Wrote {best_yearly_csv}")
    print(f"Wrote {best_monthly_csv}")
    print(f"Wrote {len(chart_outputs)} P0 chart files")
    print(f"Wrote {report_md}")
    print(f"Wrote {full_report_md}")
    print(f"Wrote {manifest_path}")
    print(
        summary.sort_values(["p0_recent_score", "last_30d_pnl_usd"], ascending=[False, False])[
            [
                "variant_id",
                "trades",
                "pnl_usd",
                "max_dd_usd",
                "return_dd",
                "march_2026_pnl_usd",
                "last_30d_trades",
                "last_30d_pnl_usd",
                "last_30d_max_dd_usd",
                "short_pnl_usd",
                "switch_count",
            ]
        ]
        .head(15)
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
