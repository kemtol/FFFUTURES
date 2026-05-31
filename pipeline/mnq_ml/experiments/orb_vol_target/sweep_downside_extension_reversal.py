#!/usr/bin/env python3
"""Sweep MNQ ORB downside-extension long reversal.

Strategy idea:
- Build the 15m opening range.
- If price extends below OR low by X * OR range, prepare a long reversal.
- Entry uses the next M1 open after the extension touch/close signal.
- Stop is one OR range below entry by default.
- Take profit is whichever is hit first: dynamic session VWAP or fixed +N R.
- Time exit remains 15:00 NY.

This is separate from the ORB continuation sweep so the long-reversal thesis
does not pollute the baseline artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Any

import pandas as pd

from common import assert_mnq_namespaces, load_config, project_path, write_json
from sweep_orb_params import hhmm_to_minutes, max_drawdown

DEFAULT_ORB_MINUTES = [15]
DEFAULT_EXTENSION_R = [0.5, 1.0, 1.5, 2.0]
DEFAULT_ENTRY_MODES = ["touch_next_open", "close_next_open"]
DEFAULT_TARGET_RISKS = [100, 200, 300, 400, 500, 600]
DEFAULT_STOP_R = [1.0]
DEFAULT_TP_R = [2.0]
WINDOW_DAYS = [5, 10, 20, 30, 50, 100, 200]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def format_r(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def format_tp_reason(tp_r: float) -> str:
    return f"TP_{format_r(tp_r)}R"


def describe_tp_grid(tp_r: list[float]) -> str:
    labels = [f"{x:g}R" for x in tp_r]
    if len(labels) == 1:
        return f"fixed +{labels[0]} from entry"
    return "fixed target grid: " + ", ".join(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orb-minutes", default=",".join(map(str, DEFAULT_ORB_MINUTES)))
    parser.add_argument("--extension-r", default=",".join(map(str, DEFAULT_EXTENSION_R)))
    parser.add_argument("--entry-modes", default=",".join(DEFAULT_ENTRY_MODES))
    parser.add_argument("--target-risks", default=",".join(map(str, DEFAULT_TARGET_RISKS)))
    parser.add_argument("--stop-r", default=",".join(map(str, DEFAULT_STOP_R)))
    parser.add_argument("--tp-r", default=",".join(map(str, DEFAULT_TP_R)))
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-max", type=float, default=None)
    parser.add_argument("--output-dir", default="data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_l1(cfg: dict[str, Any]) -> pd.DataFrame:
    l1_path = project_path(cfg["outputs"]["l1_context"])
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")
    l1 = pd.read_parquet(l1_path)
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    return l1.sort_values("timestamp_utc").reset_index(drop=True)


def add_session_vwap(day: pd.DataFrame) -> pd.DataFrame:
    out = day.copy()
    typical = (out["high"].astype(float) + out["low"].astype(float) + out["close"].astype(float)) / 3.0
    volume = out["volume"].astype(float).clip(lower=0.0)
    cum_vol = volume.cumsum()
    cum_pv = (typical * volume).cumsum()
    fallback = out["close"].astype(float).expanding(min_periods=1).mean()
    out["session_vwap"] = (cum_pv / cum_vol.replace(0.0, pd.NA)).fillna(fallback)
    return out


def add_rsi(day: pd.DataFrame, period: int) -> pd.DataFrame:
    out = day.copy()
    close = out["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out[f"rsi_{period}"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    return out


def first_signal(post: pd.DataFrame, entry_mode: str, entry_level: float) -> int | None:
    if entry_mode == "touch_next_open":
        hits = post[post["low"].astype(float) <= entry_level]
    elif entry_mode == "close_next_open":
        hits = post[post["close"].astype(float) <= entry_level]
    else:
        raise ValueError(f"Unsupported entry mode: {entry_mode}")
    if hits.empty:
        return None
    return int(hits.index[0])


def simulate_exit(
    day: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    tp_reason: str,
    time_exit: str,
    slippage_pts: float,
) -> dict[str, Any] | None:
    exit_scan = day.iloc[entry_idx:].copy()
    exit_scan = exit_scan[exit_scan["bar_data_quality_ok"].astype(bool)]
    if exit_scan.empty:
        return None
    time_exit_candidates = exit_scan[exit_scan["ny_time"] >= time_exit]
    if time_exit_candidates.empty:
        return None
    time_exit_idx = int(time_exit_candidates.iloc[0].name)
    pre_time_exit = exit_scan.loc[exit_scan.index <= time_exit_idx]

    for _, bar in pre_time_exit.iterrows():
        low = float(bar["low"])
        high = float(bar["high"])
        vwap = float(bar["session_vwap"])

        # Conservative same-bar ambiguity: if stop and profit are both touched,
        # count the stop first.
        if low <= stop_price:
            exit_price = stop_price - slippage_pts
            return {
                "exit_ts": bar["timestamp_utc"],
                "exit_reason": "SL",
                "exit_price": exit_price,
            }

        profit_candidates: list[tuple[str, float]] = []
        if vwap > entry_price and high >= vwap:
            profit_candidates.append(("VWAP", vwap))
        if high >= tp_price:
            profit_candidates.append((tp_reason, tp_price))
        if profit_candidates:
            reason, raw_exit = min(profit_candidates, key=lambda x: x[1])
            exit_price = raw_exit - slippage_pts
            return {
                "exit_ts": bar["timestamp_utc"],
                "exit_reason": reason,
                "exit_price": exit_price,
            }

    time_exit_bar = time_exit_candidates.iloc[0]
    return {
        "exit_ts": time_exit_bar["timestamp_utc"],
        "exit_reason": "TIME_EXIT",
        "exit_price": float(time_exit_bar["close"]) - slippage_pts,
    }


def build_base_opportunities(
    l1: pd.DataFrame,
    cfg: dict[str, Any],
    orb_minutes_grid: list[int],
    extension_grid: list[float],
    entry_modes: list[str],
    stop_r_grid: list[float],
    tp_r_grid: list[float],
    rsi_period: int,
    rsi_max: float | None,
) -> pd.DataFrame:
    rules = cfg["rules"]
    costs = cfg["costs"]
    session = cfg["session"]

    market_open_min = hhmm_to_minutes(session["market_open"])
    time_exit = session["time_exit"]
    time_exit_min = hhmm_to_minutes(time_exit)
    point_value = float(costs["point_value_usd"])
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    min_orb_range = float(rules["min_orb_range_pts"])
    max_orb_range = float(rules["max_orb_range_pts"])
    min_entry_risk = float(rules["min_entry_risk_pts"])
    max_entry_risk = float(rules["max_entry_risk_pts"])

    rows: list[dict[str, Any]] = []
    for ny_date, raw_day in l1.groupby("ny_date", sort=True):
        day = add_rsi(add_session_vwap(raw_day.reset_index(drop=True)), rsi_period)
        quality_mask = day["bar_data_quality_ok"].astype(bool)
        for orb_minutes in orb_minutes_grid:
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

            for extension_r in extension_grid:
                entry_level = orb_low - extension_r * orb_range
                for entry_mode in entry_modes:
                    signal_idx = first_signal(post, entry_mode, entry_level)
                    if signal_idx is None:
                        continue
                    entry_idx = signal_idx + 1
                    if entry_idx >= len(day):
                        continue
                    signal = day.iloc[signal_idx]
                    signal_rsi = float(signal[f"rsi_{rsi_period}"])
                    if rsi_max is not None and signal_rsi > rsi_max:
                        continue
                    entry_bar = day.iloc[entry_idx]
                    if (not bool(entry_bar["bar_data_quality_ok"])) or entry_bar["ny_time"] > time_exit:
                        continue

                    entry_price = float(entry_bar["open"]) + slippage_pts
                    for stop_r in stop_r_grid:
                        entry_risk_pts = stop_r * orb_range
                        if entry_risk_pts < min_entry_risk or entry_risk_pts > max_entry_risk:
                            continue
                        stop_price = entry_price - entry_risk_pts
                        for tp_r in tp_r_grid:
                            tp_price = entry_price + tp_r * entry_risk_pts
                            exit_info = simulate_exit(
                                day=day,
                                entry_idx=entry_idx,
                                entry_price=entry_price,
                                stop_price=stop_price,
                                tp_price=tp_price,
                                tp_reason=format_tp_reason(tp_r),
                                time_exit=time_exit,
                                slippage_pts=slippage_pts,
                            )
                            if exit_info is None:
                                continue

                            exit_price = float(exit_info["exit_price"])
                            gross_pts = exit_price - entry_price
                            pnl_per_contract = gross_pts * point_value
                            risk_per_contract = entry_risk_pts * point_value
                            base_event_id = (
                                f"MNQ_ORB_DOWNREV_{orb_minutes}m_ext{extension_r:g}_"
                                f"{entry_mode}_stop{stop_r:g}_tp{tp_r:g}_{ny_date}"
                            )
                            rows.append(
                                {
                                    "base_event_id": base_event_id,
                                    "ny_date": ny_date,
                                    "orb_minutes": int(orb_minutes),
                                    "extension_r": float(extension_r),
                                    "entry_mode": entry_mode,
                                    "stop_r": float(stop_r),
                                    "tp_r": float(tp_r),
                                    "side": "LONG",
                                    "signal_ts": signal["timestamp_utc"],
                                    "entry_ts": entry_bar["timestamp_utc"],
                                    "exit_ts": exit_info["exit_ts"],
                                    "exit_reason": exit_info["exit_reason"],
                                    "signal_minutes_from_open": int(signal["minutes_from_open"]),
                                    "orb_high": orb_high,
                                    "orb_low": orb_low,
                                    "orb_range_pts": orb_range,
                                    "entry_level": entry_level,
                                    "signal_rsi": signal_rsi,
                                    "rsi_period": int(rsi_period),
                                    "rsi_max": None if rsi_max is None else float(rsi_max),
                                    "entry_price": entry_price,
                                    "stop_price": stop_price,
                                    "tp_price": tp_price,
                                    "entry_vwap": float(entry_bar["session_vwap"]),
                                    "exit_price": exit_price,
                                    "entry_risk_pts": entry_risk_pts,
                                    "risk_per_contract_usd": risk_per_contract,
                                    "pnl_per_contract_usd": pnl_per_contract,
                                    "r_multiple": pnl_per_contract / risk_per_contract if risk_per_contract else 0.0,
                                    "label": int(pnl_per_contract > 0),
                                }
                            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["orb_minutes", "extension_r", "entry_mode", "signal_ts"]).reset_index(drop=True)


def expand_target_risks(base: pd.DataFrame, cfg: dict[str, Any], target_risks: list[int]) -> pd.DataFrame:
    if base.empty:
        return base.copy()
    max_contracts = int(cfg["position_sizing"]["max_contracts"])
    min_contracts = int(cfg["position_sizing"]["min_contracts"])
    frames = []
    for risk in target_risks:
        cur = base.copy()
        cur["target_risk_usd"] = int(risk)
        cur["contracts_float"] = cur["target_risk_usd"] / cur["risk_per_contract_usd"]
        cur["contracts_floor"] = cur["contracts_float"].apply(floor).astype(int)
        cur["contracts_used"] = cur["contracts_floor"].clip(lower=0, upper=max_contracts)
        cur = cur[cur["contracts_used"] >= min_contracts].copy()
        cur["pnl_usd"] = cur["pnl_per_contract_usd"] * cur["contracts_used"]
        cur["event_id"] = cur["base_event_id"] + "_risk" + cur["target_risk_usd"].astype(str)
        frames.append(cur)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["orb_minutes", "extension_r", "entry_mode", "stop_r", "tp_r", "target_risk_usd", "signal_ts"]
    )


def summarize_group(group: pd.DataFrame, anchor: pd.Timestamp) -> dict[str, Any]:
    group = group.sort_values("signal_ts")
    total_pnl = float(group["pnl_usd"].sum())
    max_dd = max_drawdown(group["pnl_usd"])
    result = {
        "orb_minutes": int(group["orb_minutes"].iloc[0]),
        "extension_r": float(group["extension_r"].iloc[0]),
        "entry_mode": str(group["entry_mode"].iloc[0]),
        "stop_r": float(group["stop_r"].iloc[0]),
        "tp_r": float(group["tp_r"].iloc[0]),
        "target_risk_usd": int(group["target_risk_usd"].iloc[0]),
        "rsi_period": int(group["rsi_period"].iloc[0]),
        "rsi_max": None if pd.isna(group["rsi_max"].iloc[0]) else float(group["rsi_max"].iloc[0]),
        "trades": int(len(group)),
        "win_rate": float(group["label"].mean()),
        "total_pnl_usd": total_pnl,
        "avg_pnl_usd": float(group["pnl_usd"].mean()),
        "avg_r_multiple": float(group["r_multiple"].mean()),
        "max_dd_usd": max_dd,
        "return_dd": total_pnl / abs(max_dd) if max_dd < 0 else None,
        "avg_contracts": float(group["contracts_used"].mean()),
        "min_signal_ts": group["signal_ts"].min().isoformat(),
        "max_signal_ts": group["signal_ts"].max().isoformat(),
    }
    result.update({f"exit_{k.lower()}": int(v) for k, v in group["exit_reason"].value_counts().to_dict().items()})

    y2026 = group[group["signal_ts"].dt.year == 2026]
    result.update(
        {
            "trades_2026": int(len(y2026)),
            "win_rate_2026": float(y2026["label"].mean()) if not y2026.empty else 0.0,
            "pnl_2026_usd": float(y2026["pnl_usd"].sum()) if not y2026.empty else 0.0,
            "max_dd_2026_usd": max_drawdown(y2026["pnl_usd"]) if not y2026.empty else 0.0,
        }
    )

    for days in WINDOW_DAYS:
        start = anchor - pd.Timedelta(days=days)
        window = group[(group["signal_ts"] > start) & (group["signal_ts"] <= anchor)]
        prefix = f"w{days}d"
        result[f"{prefix}_trades"] = int(len(window))
        result[f"{prefix}_win_rate"] = float(window["label"].mean()) if not window.empty else 0.0
        result[f"{prefix}_pnl_usd"] = float(window["pnl_usd"].sum()) if not window.empty else 0.0
        result[f"{prefix}_max_dd_usd"] = max_drawdown(window["pnl_usd"]) if not window.empty else 0.0

    return result


def build_results(events: pd.DataFrame, anchor: pd.Timestamp, min_trades: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["orb_minutes", "extension_r", "entry_mode", "stop_r", "tp_r", "target_risk_usd"]
    for _, group in events.groupby(group_cols, sort=True):
        row = summarize_group(group, anchor)
        row["meets_min_trades"] = bool(row["trades"] >= min_trades)
        rows.append(row)
    results = pd.DataFrame(rows)
    results["rank_return_dd"] = results["return_dd"].rank(ascending=False, method="min")
    results["rank_2026_pnl"] = results["pnl_2026_usd"].rank(ascending=False, method="min")
    results["rank_recent_50d_pnl"] = results["w50d_pnl_usd"].rank(ascending=False, method="min")
    results["score_quick"] = (
        results["return_dd"].fillna(0.0)
        + (results["pnl_2026_usd"] / 3000.0)
        + (results["w50d_pnl_usd"] / 3000.0)
    )
    return results.sort_values(
        ["meets_min_trades", "score_quick", "return_dd", "pnl_2026_usd"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def write_report(path: Path, results: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# MNQ ORB Downside Extension Long Reversal Sweep",
        "",
        f"Created: `{manifest['created_at']}`",
        "",
        "## Contract",
        "",
        "- Long-only reversal after downside extension below the 15m opening range.",
        "- Entry modes: `touch_next_open` and `close_next_open`.",
        f"- TP is whichever is hit first: dynamic session VWAP or {manifest['tp_description']}.",
        "- SL is one OR range below entry for the default sweep.",
        "- Same-bar SL/TP ambiguity is resolved conservatively: SL first.",
        f"- RSI filter: `{manifest['rsi_filter']}`.",
        "",
        "## Top Rows",
        "",
        "| Rank | Ext R | Entry | RSI Max | Risk | Trades | Win Rate | PnL | Max DD | R/DD | 5D | 10D | 20D | 50D | 100D | 200D |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    top = results.head(20).copy()
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        ret_dd = "" if pd.isna(row.return_dd) else f"{row.return_dd:.2f}"
        rsi_label = "" if pd.isna(row.rsi_max) else f"{row.rsi_max:g}"
        lines.append(
            f"| {idx} | {row.extension_r:g} | `{row.entry_mode}` | {rsi_label} | ${row.target_risk_usd} | "
            f"{row.trades} | {row.win_rate:.1%} | ${row.total_pnl_usd:,.0f} | ${row.max_dd_usd:,.0f} | {ret_dd} | "
            f"${row.w5d_pnl_usd:,.0f} / ${row.w5d_max_dd_usd:,.0f} | "
            f"${row.w10d_pnl_usd:,.0f} / ${row.w10d_max_dd_usd:,.0f} | "
            f"${row.w20d_pnl_usd:,.0f} / ${row.w20d_max_dd_usd:,.0f} | "
            f"${row.w50d_pnl_usd:,.0f} / ${row.w50d_max_dd_usd:,.0f} | "
            f"${row.w100d_pnl_usd:,.0f} / ${row.w100d_max_dd_usd:,.0f} | "
            f"${row.w200d_pnl_usd:,.0f} / ${row.w200d_max_dd_usd:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This is a first-pass research sweep, not a live candidate.",
            "- The entry assumption is next-open after the extension signal, not guaranteed intrabar limit fill.",
            "- Compare against the main long-only ORB continuation before adding ML.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    cfg = load_config()
    assert_mnq_namespaces(cfg)
    orb_minutes = parse_int_list(args.orb_minutes)
    extension_r = parse_float_list(args.extension_r)
    entry_modes = parse_str_list(args.entry_modes)
    target_risks = parse_int_list(args.target_risks)
    stop_r = parse_float_list(args.stop_r)
    tp_r = parse_float_list(args.tp_r)
    invalid_entry_modes = sorted(set(entry_modes) - {"touch_next_open", "close_next_open"})
    if invalid_entry_modes:
        raise SystemExit(f"Invalid entry modes: {invalid_entry_modes}")

    output_dir = project_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output dir exists; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "downside_reversal_base_opportunities.parquet"
    events_path = output_dir / "downside_reversal_events.parquet"
    results_path = output_dir / "downside_reversal_results.parquet"
    manifest_path = output_dir / "downside_reversal_manifest.json"
    report_path = output_dir / "README.md"

    l1 = load_l1(cfg)
    anchor = pd.Timestamp(l1["timestamp_utc"].max())
    base = build_base_opportunities(
        l1=l1,
        cfg=cfg,
        orb_minutes_grid=orb_minutes,
        extension_grid=extension_r,
        entry_modes=entry_modes,
        stop_r_grid=stop_r,
        tp_r_grid=tp_r,
        rsi_period=args.rsi_period,
        rsi_max=args.rsi_max,
    )
    events = expand_target_risks(base, cfg, target_risks)
    results = build_results(events, anchor, args.min_trades)

    base.to_parquet(base_path, index=False)
    events.to_parquet(events_path, index=False)
    results.to_parquet(results_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "mnq_orb_downside_extension_long_reversal",
        "l1_context": cfg["outputs"]["l1_context"],
        "anchor_ts": anchor.isoformat(),
        "orb_minutes": orb_minutes,
        "extension_r": extension_r,
        "entry_modes": entry_modes,
        "target_risks": target_risks,
        "stop_r": stop_r,
        "tp_r": tp_r,
        "tp_description": describe_tp_grid(tp_r),
        "rsi_period": int(args.rsi_period),
        "rsi_max": args.rsi_max,
        "rsi_filter": "disabled" if args.rsi_max is None else f"RSI{args.rsi_period} <= {args.rsi_max:g}",
        "base_rows": int(len(base)),
        "event_rows": int(len(events)),
        "result_rows": int(len(results)),
        "outputs": {
            "base_opportunities": str(base_path),
            "events": str(events_path),
            "results": str(results_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
    }
    write_json(manifest_path, manifest)
    write_report(report_path, results, manifest)
    print(json.dumps({"status": "PASS", **manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
