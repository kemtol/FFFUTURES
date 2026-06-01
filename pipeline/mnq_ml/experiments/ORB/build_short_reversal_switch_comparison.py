#!/usr/bin/env python3
"""Compare NASDAQ ORB short breakout variants with switch-to-long logic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from build_supertrend_regime_features import EVENTS_PATH, MODEL_DIR, load_anchor, max_drawdown, profit_factor
from common import assert_mnq_namespaces, load_config, project_path, write_json
from sweep_orb_params import hhmm_to_minutes, load_l1, minutes_to_hhmm

SWEEP_EVENTS_PATH = "data/Level_2_Datamart/mnq/ORB/sweeps/sweep_events.parquet"
OUTPUT_DIR = "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod"
MODEL_OUTPUT_DIR = "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
EVENTS_CSV = "short_reversal_switch_events.csv"
LEGS_CSV = "short_reversal_switch_legs.csv"
SUMMARY_CSV = "short_reversal_switch_comparison.csv"
REPORT_MD = "short_reversal_switch_comparison.md"
MANIFEST_JSON = "short_reversal_switch_comparison_manifest.json"
WINDOW_DAYS = [5, 10, 20, 30, 50, 100, 200]
SHORT_TP_R_VALUES = [1.0, 1.5, 2.0]
COLORS = {
    "long_only_no_st": "#334155",
    "long_short_first_breakout_no_switch": "#dc2626",
    "short_switch_tp1r": "#0f766e",
    "short_switch_tp1_5r": "#2563eb",
    "short_switch_tp2r": "#7c3aed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--orb-minutes", type=int, default=15)
    parser.add_argument("--target-risk", type=float, default=500.0)
    parser.add_argument("--long-tp-r", type=float, default=2.0)
    parser.add_argument("--short-tp-r-values", default="1,1.5,2")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def variant_id_for_tp(tp_r: float) -> str:
    raw = f"{tp_r:g}".replace(".", "_")
    return f"short_switch_tp{raw}r"


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


def load_baseline_events() -> pd.DataFrame:
    df = pd.read_parquet(project_path(EVENTS_PATH)).copy()
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    df["variant_id"] = "long_only_no_st"
    df["variant_label"] = "Long only, no ST"
    df["pnl_usd"] = df["pnl_net_usd"].astype(float)
    df["first_side"] = "LONG"
    df["switched_to_long"] = False
    df["leg_count"] = 1
    df["short_tp_r"] = pd.NA
    return df.sort_values("signal_ts").reset_index(drop=True)


def load_legacy_long_short(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(project_path(SWEEP_EVENTS_PATH)).copy()
    mask = (
        df["orb_minutes"].eq(args.orb_minutes)
        & df["side_mode"].eq("long_short")
        & df["exit_mode"].eq("tp_2r_or_time")
        & df["target_risk_usd"].astype(float).eq(float(args.target_risk))
    )
    out = df[mask].copy()
    if out.empty:
        raise SystemExit("Missing legacy long_short sweep events.")
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        out[col] = pd.to_datetime(out[col], utc=True)
    out["variant_id"] = "long_short_first_breakout_no_switch"
    out["variant_label"] = "Long+Short first breakout, no switch"
    out["pnl_net_usd"] = out["pnl_usd"].astype(float)
    out["first_side"] = out["side"]
    out["switched_to_long"] = False
    out["leg_count"] = 1
    out["short_tp_r"] = 2.0
    return out.sort_values("signal_ts").reset_index(drop=True)


def size_contracts(entry_risk_pts: float, cfg: dict, target_risk: float) -> tuple[int, float, float]:
    point_value = float(cfg["costs"]["point_value_usd"])
    max_contracts = int(cfg["position_sizing"]["max_contracts"])
    min_contracts = int(cfg["position_sizing"]["min_contracts"])
    risk_per_contract = entry_risk_pts * point_value
    if risk_per_contract <= 0:
        return 0, risk_per_contract, 0.0
    contracts_float = target_risk / risk_per_contract
    contracts = floor(contracts_float)
    contracts = min(max_contracts, contracts)
    if contracts < min_contracts:
        return 0, risk_per_contract, contracts_float
    return contracts, risk_per_contract, contracts_float


def build_leg(
    *,
    variant_id: str,
    ny_date: str,
    sequence_id: str,
    leg_index: int,
    side: str,
    signal_bar: pd.Series,
    entry_bar: pd.Series,
    exit_bar: pd.Series,
    exit_reason: str,
    raw_exit_price: float,
    entry_price: float,
    stop_reference: float,
    entry_risk_pts: float,
    contracts: int,
    risk_per_contract: float,
    contracts_float: float,
    cfg: dict,
    target_risk: float,
    orb_high: float,
    orb_low: float,
    orb_range: float,
    orb_minutes: int,
    short_tp_r: float,
) -> dict[str, Any]:
    costs = cfg["costs"]
    point_value = float(costs["point_value_usd"])
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    commission = float(costs["commission_round_turn_usd"])

    if side == "LONG":
        exit_price = float(raw_exit_price) - slippage_pts
        gross_pts = exit_price - entry_price
    else:
        exit_price = float(raw_exit_price) + slippage_pts
        gross_pts = entry_price - exit_price

    pnl_per_contract = gross_pts * point_value - commission
    pnl_usd = pnl_per_contract * contracts
    return {
        "variant_id": variant_id,
        "sequence_id": sequence_id,
        "ny_date": ny_date,
        "leg_index": leg_index,
        "side": side,
        "orb_minutes": orb_minutes,
        "orb_end": minutes_to_hhmm(hhmm_to_minutes(cfg["session"]["market_open"]) + orb_minutes),
        "short_tp_r": short_tp_r,
        "signal_ts": signal_bar["timestamp_utc"],
        "entry_ts": entry_bar["timestamp_utc"],
        "exit_ts": exit_bar["timestamp_utc"],
        "signal_minutes_from_open": int(signal_bar["minutes_from_open"]),
        "exit_reason": exit_reason,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_range_pts": orb_range,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_reference": stop_reference,
        "entry_risk_pts": entry_risk_pts,
        "risk_per_contract_usd": risk_per_contract,
        "target_risk_usd": target_risk,
        "contracts_float": contracts_float,
        "contracts_used": contracts,
        "pnl_per_contract_usd": pnl_per_contract,
        "pnl_usd": pnl_usd,
        "label": int(pnl_usd > 0),
    }


def scan_leg_exit(
    *,
    side: str,
    day: pd.DataFrame,
    signal_bar: pd.Series,
    entry_idx: int,
    entry_price: float,
    entry_risk_pts: float,
    tp_r: float,
    orb_high: float,
    time_exit: str,
    allow_short_switch: bool,
) -> tuple[pd.Series, str, float, int | None, pd.Series | None]:
    exit_scan = day.iloc[entry_idx:]
    exit_scan = exit_scan[exit_scan["bar_data_quality_ok"].astype(bool)]
    time_exit_candidates = exit_scan[exit_scan["ny_time"] >= time_exit]
    if time_exit_candidates.empty:
        raise ValueError("missing time exit")
    time_exit_idx = int(time_exit_candidates.iloc[0].name)
    tp_price = entry_price + tp_r * entry_risk_pts if side == "LONG" else entry_price - tp_r * entry_risk_pts

    for idx, row in exit_scan.iterrows():
        if idx > time_exit_idx:
            break
        if side == "LONG" and float(row["high"]) >= tp_price:
            return row, f"TP_{tp_r:g}R", tp_price, None, None
        if side == "SHORT" and float(row["low"]) <= tp_price:
            return row, f"TP_{tp_r:g}R", tp_price, None, None
        if idx == time_exit_idx or row["ny_time"] >= time_exit:
            return row, "TIME_EXIT", float(row["close"]), None, None
        if allow_short_switch and side == "SHORT" and float(row["close"]) > orb_high:
            switch_entry_idx = idx + 1
            if switch_entry_idx < len(day):
                switch_entry_bar = day.iloc[switch_entry_idx]
                if bool(switch_entry_bar["bar_data_quality_ok"]) and switch_entry_bar["ny_time"] <= time_exit:
                    return switch_entry_bar, "SWITCH_TO_LONG", float(switch_entry_bar["open"]), switch_entry_idx, row
    time_exit_bar = time_exit_candidates.iloc[0]
    return time_exit_bar, "TIME_EXIT", float(time_exit_bar["close"]), None, None


def simulate_switch_variant(
    l1: pd.DataFrame,
    cfg: dict,
    *,
    orb_minutes: int,
    target_risk: float,
    short_tp_r: float,
    long_tp_r: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = cfg["rules"]
    costs = cfg["costs"]
    session = cfg["session"]
    market_open_min = hhmm_to_minutes(session["market_open"])
    time_exit = session["time_exit"]
    time_exit_min = hhmm_to_minutes(time_exit)
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    min_orb_range = float(rules["min_orb_range_pts"])
    max_orb_range = float(rules["max_orb_range_pts"])
    min_entry_risk = float(rules["min_entry_risk_pts"])
    max_entry_risk = float(rules["max_entry_risk_pts"])
    variant_id = variant_id_for_tp(short_tp_r)
    variant_label = f"Short switch to long, short TP {short_tp_r:g}R"

    sequences = []
    legs = []
    for ny_date, day in l1.groupby("ny_date", sort=True):
        day = day.reset_index(drop=True)
        quality_mask = day["bar_data_quality_ok"].astype(bool)
        orb_mask = quality_mask & (day["minutes_from_open"] > 0) & (day["minutes_from_open"] <= orb_minutes)
        orb = day.loc[orb_mask]
        if len(orb) != orb_minutes:
            continue
        orb_high = float(orb["high"].max())
        orb_low = float(orb["low"].min())
        orb_range = orb_high - orb_low
        if orb_range < min_orb_range or orb_range > max_orb_range:
            continue

        post = day[
            quality_mask
            & (day["minutes_from_open"] > orb_minutes)
            & (day["minutes_from_open"] < (time_exit_min - market_open_min))
        ]
        if post.empty:
            continue
        first_candidates: dict[str, int] = {}
        long_candidates = post[post["close"] > orb_high]
        short_candidates = post[post["close"] < orb_low]
        if not long_candidates.empty:
            first_candidates["LONG"] = int(long_candidates.index[0])
        if not short_candidates.empty:
            first_candidates["SHORT"] = int(short_candidates.index[0])
        if not first_candidates:
            continue
        first_side, signal_idx = sorted(first_candidates.items(), key=lambda x: x[1])[0]
        entry_idx = signal_idx + 1
        if entry_idx >= len(day):
            continue
        signal_bar = day.iloc[signal_idx]
        entry_bar = day.iloc[entry_idx]
        if (not bool(entry_bar["bar_data_quality_ok"])) or entry_bar["ny_time"] > time_exit:
            continue

        sequence_id = f"NASDAQ_ORB_{orb_minutes}m_short_switch_tp{short_tp_r:g}_{ny_date}"
        leg_rows = []
        try:
            if first_side == "LONG":
                entry_price = float(entry_bar["open"]) + slippage_pts
                stop_reference = orb_low
                entry_risk_pts = entry_price - stop_reference
                if entry_risk_pts < min_entry_risk or entry_risk_pts > max_entry_risk:
                    continue
                contracts, risk_per_contract, contracts_float = size_contracts(entry_risk_pts, cfg, target_risk)
                if contracts <= 0:
                    continue
                exit_bar, exit_reason, raw_exit_price, _, _ = scan_leg_exit(
                    side="LONG",
                    day=day,
                    signal_bar=signal_bar,
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    entry_risk_pts=entry_risk_pts,
                    tp_r=long_tp_r,
                    orb_high=orb_high,
                    time_exit=time_exit,
                    allow_short_switch=False,
                )
                leg_rows.append(
                    build_leg(
                        variant_id=variant_id,
                        ny_date=ny_date,
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
                        target_risk=target_risk,
                        orb_high=orb_high,
                        orb_low=orb_low,
                        orb_range=orb_range,
                        orb_minutes=orb_minutes,
                        short_tp_r=short_tp_r,
                    )
                )
            else:
                entry_price = float(entry_bar["open"]) - slippage_pts
                stop_reference = orb_high
                entry_risk_pts = stop_reference - entry_price
                if entry_risk_pts < min_entry_risk or entry_risk_pts > max_entry_risk:
                    continue
                contracts, risk_per_contract, contracts_float = size_contracts(entry_risk_pts, cfg, target_risk)
                if contracts <= 0:
                    continue
                exit_bar, exit_reason, raw_exit_price, switch_entry_idx, switch_signal_bar = scan_leg_exit(
                    side="SHORT",
                    day=day,
                    signal_bar=signal_bar,
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    entry_risk_pts=entry_risk_pts,
                    tp_r=short_tp_r,
                    orb_high=orb_high,
                    time_exit=time_exit,
                    allow_short_switch=True,
                )
                leg_rows.append(
                    build_leg(
                        variant_id=variant_id,
                        ny_date=ny_date,
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
                        target_risk=target_risk,
                        orb_high=orb_high,
                        orb_low=orb_low,
                        orb_range=orb_range,
                        orb_minutes=orb_minutes,
                        short_tp_r=short_tp_r,
                    )
                )
                if switch_entry_idx is not None and switch_signal_bar is not None:
                    long_entry_bar = day.iloc[switch_entry_idx]
                    long_entry_price = float(long_entry_bar["open"]) + slippage_pts
                    long_stop_reference = orb_low
                    long_entry_risk_pts = long_entry_price - long_stop_reference
                    if min_entry_risk <= long_entry_risk_pts <= max_entry_risk:
                        long_contracts, long_risk_per_contract, long_contracts_float = size_contracts(
                            long_entry_risk_pts, cfg, target_risk
                        )
                        if long_contracts > 0:
                            long_exit_bar, long_exit_reason, long_raw_exit_price, _, _ = scan_leg_exit(
                                side="LONG",
                                day=day,
                                signal_bar=switch_signal_bar,
                                entry_idx=switch_entry_idx,
                                entry_price=long_entry_price,
                                entry_risk_pts=long_entry_risk_pts,
                                tp_r=long_tp_r,
                                orb_high=orb_high,
                                time_exit=time_exit,
                                allow_short_switch=False,
                            )
                            leg_rows.append(
                                build_leg(
                                    variant_id=variant_id,
                                    ny_date=ny_date,
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
                                    target_risk=target_risk,
                                    orb_high=orb_high,
                                    orb_low=orb_low,
                                    orb_range=orb_range,
                                    orb_minutes=orb_minutes,
                                    short_tp_r=short_tp_r,
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
        sequences.append(
            {
                "variant_id": variant_id,
                "variant_label": variant_label,
                "sequence_id": sequence_id,
                "ny_date": ny_date,
                "orb_minutes": orb_minutes,
                "short_tp_r": short_tp_r,
                "long_tp_r": long_tp_r,
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
        leg_df = leg_df.sort_values(["variant_id", "signal_ts", "leg_index"]).reset_index(drop=True)
    return seq_df, leg_df


def subset_dates(events: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(events["ny_date"])
    return events[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def summarize(events: pd.DataFrame, anchor: pd.Timestamp, variant_id: str, label: str) -> dict[str, Any]:
    pnl = events["pnl_net_usd"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total = float(pnl.sum()) if not pnl.empty else 0.0
    dd = max_drawdown(pnl)
    row = {
        "variant_id": variant_id,
        "label": label,
        "trades": int(len(events)),
        "win_rate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
        "pnl_usd": total,
        "max_dd_usd": dd,
        "return_dd": total / abs(dd) if dd else None,
        "profit_factor": profit_factor(pnl),
        "avg_trade_usd": float(pnl.mean()) if not pnl.empty else 0.0,
        "long_first_trades": int(events["first_side"].eq("LONG").sum()) if "first_side" in events else int(len(events)),
        "short_first_trades": int(events["first_side"].eq("SHORT").sum()) if "first_side" in events else 0,
        "switch_count": int(events["switched_to_long"].fillna(False).sum()) if "switched_to_long" in events else 0,
        "long_pnl_usd": float(events.get("long_pnl_usd", pd.Series(0.0, index=events.index)).sum()),
        "short_pnl_usd": float(events.get("short_pnl_usd", pd.Series(0.0, index=events.index)).sum()),
        "gross_profit_usd": float(wins.sum()) if not wins.empty else 0.0,
        "gross_loss_usd": float(losses.sum()) if not losses.empty else 0.0,
    }
    for prefix, subset in [
        ("jan_may_2026", subset_dates(events, "2026-01-01", "2026-05-31")),
        ("march_2026", subset_dates(events, "2026-03-01", "2026-03-31")),
    ]:
        sub_pnl = subset["pnl_net_usd"].astype(float)
        row[f"{prefix}_trades"] = int(len(subset))
        row[f"{prefix}_pnl_usd"] = float(sub_pnl.sum()) if not sub_pnl.empty else 0.0
        row[f"{prefix}_max_dd_usd"] = max_drawdown(sub_pnl) if not sub_pnl.empty else 0.0
    for days in WINDOW_DAYS:
        window = events[(events["signal_ts"] > anchor - pd.Timedelta(days=days)) & (events["signal_ts"] <= anchor)]
        w_pnl = window["pnl_net_usd"].astype(float)
        row[f"last_{days}d_trades"] = int(len(window))
        row[f"last_{days}d_pnl_usd"] = float(w_pnl.sum()) if not w_pnl.empty else 0.0
        row[f"last_{days}d_max_dd_usd"] = max_drawdown(w_pnl) if not w_pnl.empty else 0.0
    return row


def build_summary(all_events: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for variant_id, group in all_events.groupby("variant_id", sort=False):
        label = str(group["variant_label"].iloc[0])
        rows.append(summarize(group.sort_values("signal_ts"), anchor, variant_id, label))
    return pd.DataFrame(rows)


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


def summary_table(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "| {label} | {trades:,} | {wr} | {pnl} | {dd} | {retdd} | {shorts:,} | {switches:,} | {short_pnl} | {jm_pnl} | {mar_pnl} | {d30_pnl} | {d30_dd} |".format(
                label=row["label"],
                trades=int(row["trades"]),
                wr=pct(row["win_rate"]),
                pnl=money(row["pnl_usd"]),
                dd=money(row["max_dd_usd"]),
                retdd=number(row["return_dd"]),
                shorts=int(row["short_first_trades"]),
                switches=int(row["switch_count"]),
                short_pnl=money(row["short_pnl_usd"]),
                jm_pnl=money(row["jan_may_2026_pnl_usd"]),
                mar_pnl=money(row["march_2026_pnl_usd"]),
                d30_pnl=money(row["last_30d_pnl_usd"]),
                d30_dd=money(row["last_30d_max_dd_usd"]),
            )
        )
    return "\n".join(rows)


def last30_detail_table(events: pd.DataFrame, variant_id: str, anchor: pd.Timestamp) -> str:
    window = events[
        (events["variant_id"].eq(variant_id))
        & (events["signal_ts"] > anchor - pd.Timedelta(days=30))
        & (events["signal_ts"] <= anchor)
    ].copy()
    if window.empty:
        return ""
    rows = []
    for _, row in window.sort_values("signal_ts").iterrows():
        rows.append(
            "| {date} | {first} | {switched} | {legs} | {reason} | {pnl} | {short_pnl} | {long_pnl} |".format(
                date=row["ny_date"],
                first=row["first_side"],
                switched="Yes" if bool(row.get("switched_to_long", False)) else "No",
                legs=int(row.get("leg_count", 1)),
                reason=row["exit_reason"],
                pnl=money(row["pnl_net_usd"]),
                short_pnl=money(row.get("short_pnl_usd", 0.0)),
                long_pnl=money(row.get("long_pnl_usd", 0.0)),
            )
        )
    return "\n".join(rows)


def plot_equity(all_events: pd.DataFrame, chart_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for variant_id, group in all_events.groupby("variant_id", sort=False):
        equity = group.sort_values("signal_ts")["pnl_net_usd"].astype(float).cumsum()
        ax.plot(group.sort_values("signal_ts")["signal_ts"], equity, label=group["variant_label"].iloc[0], color=COLORS.get(variant_id), linewidth=1.5)
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "NASDAQ ORB Short Switch Variant Equity", "Cumulative PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    files = save_fig(fig, chart_dir, "short_reversal_switch_equity_curve")
    plt.close(fig)
    return files


def plot_drawdown(all_events: pd.DataFrame, chart_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for variant_id, group in all_events.groupby("variant_id", sort=False):
        ordered = group.sort_values("signal_ts")
        equity = ordered["pnl_net_usd"].astype(float).cumsum()
        dd = equity - equity.cummax()
        ax.plot(ordered["signal_ts"], dd, label=group["variant_label"].iloc[0], color=COLORS.get(variant_id), linewidth=1.5)
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "NASDAQ ORB Short Switch Variant Drawdown", "Drawdown ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    files = save_fig(fig, chart_dir, "short_reversal_switch_drawdown_curve")
    plt.close(fig)
    return files


def plot_last30(all_events: pd.DataFrame, anchor: pd.Timestamp, chart_dir: Path) -> list[str]:
    window = all_events[
        (all_events["signal_ts"] > anchor - pd.Timedelta(days=30))
        & (all_events["signal_ts"] <= anchor)
    ].copy()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for variant_id, group in window.groupby("variant_id", sort=False):
        ordered = group.sort_values("signal_ts")
        equity = ordered["pnl_net_usd"].astype(float).cumsum()
        ax.plot(ordered["signal_ts"], equity, marker="o", label=ordered["variant_label"].iloc[0], color=COLORS.get(variant_id), linewidth=1.5)
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "NASDAQ ORB Short Switch Last 30D Equity", "30D cumulative PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    files = save_fig(fig, chart_dir, "short_reversal_switch_last30_equity")
    plt.close(fig)
    return files


def write_report(
    path: Path,
    summary: pd.DataFrame,
    all_events: pd.DataFrame,
    anchor: pd.Timestamp,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# NASDAQ ORB Short Breakout Switch-To-Long Comparison",
        "",
        "## Scope",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Instrument | NASDAQ Micro Futures (`MNQ`) |",
        "| ORB | 15m New York opening range |",
        "| Long rule | Baseline long continuation, TP 2R or 15:00 NY EOD |",
        "| Short rule | If OR low breaks first, enter short; if price closes above OR high, close short and switch to long next M1 open |",
        "| Short TP sweep | 1R, 1.5R, 2R |",
        "| Cost model | TopstepX MNQ commission + 1 tick slippage per side |",
        "| Lookahead handling | Reversal is based on M1 close; switch execution uses next M1 open |",
        f"| Anchor | {anchor.isoformat()} |",
        "",
        "## Charts",
        "",
        "### Equity Curve",
        "",
        f"![Short Switch Equity]({raw_url('charts/short_reversal_switch_equity_curve.png')})",
        "",
        "### Drawdown Curve",
        "",
        f"![Short Switch Drawdown]({raw_url('charts/short_reversal_switch_drawdown_curve.png')})",
        "",
        "### Last 30D Equity",
        "",
        f"![Short Switch Last 30D]({raw_url('charts/short_reversal_switch_last30_equity.png')})",
        "",
        "## Summary",
        "",
        "| Variant | Trades | WR | PnL | DD | Ret/DD | Short-first | Switches | Short PnL | Jan-May PnL | Mar PnL | 30D PnL | 30D DD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        summary_table(summary),
        "",
        "## Last 30D Detail - Switch Variants",
        "",
    ]
    for tp_r in SHORT_TP_R_VALUES:
        variant_id = variant_id_for_tp(tp_r)
        lines.extend(
            [
                f"### Short TP {tp_r:g}R",
                "",
                "| NY Date | First Side | Switched | Legs | Exit Path | PnL | Short PnL | Long PnL |",
                "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
                last30_detail_table(all_events, variant_id, anchor),
                "",
            ]
        )
    lines.extend(
        [
            "## Current Read",
            "",
            "- This test matches the intended asymmetric NASDAQ logic: short is allowed, but a failed short must yield to long continuation.",
            "- The key comparison is not whether short can trade more often, but whether the short leg improves drawdown without stealing the natural long bias.",
            "- If all short TP variants still degrade 30D or drawdown, short should remain a separate exploratory module rather than part of the primary ORB.",
            "",
            "## Artifacts",
            "",
            "| Artifact | Path |",
            "| --- | --- |",
            f"| Summary CSV | `{manifest['artifacts']['summary_csv']}` |",
            f"| Sequence events CSV | `{manifest['artifacts']['events_csv']}` |",
            f"| Leg-level CSV | `{manifest['artifacts']['legs_csv']}` |",
            f"| Manifest | `{manifest['artifacts']['manifest']}` |",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg = load_config()
    assert_mnq_namespaces(cfg)
    short_tp_values = parse_float_list(args.short_tp_r_values)
    global SHORT_TP_R_VALUES
    SHORT_TP_R_VALUES = short_tp_values

    data_dir = project_path(OUTPUT_DIR)
    model_dir = project_path(MODEL_OUTPUT_DIR)
    chart_dir = model_dir / "charts"
    summary_csv = model_dir / SUMMARY_CSV
    events_csv = model_dir / EVENTS_CSV
    legs_csv = model_dir / LEGS_CSV
    report_md = model_dir / REPORT_MD
    manifest_path = data_dir / MANIFEST_JSON
    for path in [summary_csv, events_csv, legs_csv, report_md, manifest_path]:
        if path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing artifact without --force: {path}")

    l1 = load_l1(cfg)
    baseline = load_baseline_events()
    anchor = load_anchor(baseline)
    legacy = load_legacy_long_short(args)

    switch_events = []
    switch_legs = []
    for short_tp_r in short_tp_values:
        seq, legs = simulate_switch_variant(
            l1,
            cfg,
            orb_minutes=args.orb_minutes,
            target_risk=args.target_risk,
            short_tp_r=short_tp_r,
            long_tp_r=args.long_tp_r,
        )
        if not seq.empty:
            switch_events.append(seq)
        if not legs.empty:
            switch_legs.append(legs)

    event_frames = [baseline, legacy] + switch_events
    all_events = pd.concat(event_frames, ignore_index=True, sort=False).sort_values(["variant_id", "signal_ts"])
    all_legs = pd.concat(switch_legs, ignore_index=True, sort=False) if switch_legs else pd.DataFrame()
    summary = build_summary(all_events, anchor)

    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    all_events.to_csv(events_csv, index=False)
    all_legs.to_csv(legs_csv, index=False)
    chart_outputs = []
    chart_outputs += plot_equity(all_events, chart_dir)
    chart_outputs += plot_drawdown(all_events, chart_dir)
    chart_outputs += plot_last30(all_events, anchor, chart_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchor_ts": anchor.isoformat(),
        "params": {
            "orb_minutes": args.orb_minutes,
            "target_risk_usd": args.target_risk,
            "long_tp_r": args.long_tp_r,
            "short_tp_r_values": short_tp_values,
            "switch_rule": "short closes and reverses to long on first M1 close above OR high; execution next M1 open",
        },
        "rows": {
            "summary": int(len(summary)),
            "events": int(len(all_events)),
            "legs": int(len(all_legs)),
        },
        "artifacts": {
            "summary_csv": str(summary_csv.relative_to(project_path("."))),
            "events_csv": str(events_csv.relative_to(project_path("."))),
            "legs_csv": str(legs_csv.relative_to(project_path("."))),
            "report": str(report_md.relative_to(project_path("."))),
            "charts": chart_outputs,
            "manifest": str(manifest_path.relative_to(project_path("."))),
        },
    }
    write_json(manifest_path, manifest)
    write_report(report_md, summary, all_events, anchor, manifest)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {events_csv}")
    print(f"Wrote {legs_csv}")
    print(f"Wrote {report_md}")
    print(f"Wrote {manifest_path}")
    print(
        summary[
            [
                "label",
                "trades",
                "pnl_usd",
                "max_dd_usd",
                "return_dd",
                "short_first_trades",
                "switch_count",
                "short_pnl_usd",
                "march_2026_pnl_usd",
                "last_30d_pnl_usd",
                "last_30d_max_dd_usd",
            ]
        ].round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
