#!/usr/bin/env python3
"""Build SuperTrend regime features and filter audit for MNQ ORB baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import assert_mnq_namespaces, load_config, project_path, write_json

STRATEGY_ID = "rule_based_15m_long_tp2r_eod"
EVENTS_PATH = "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet"
FEATURES_PATH = (
    "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/"
    "supertrend_regime_features.parquet"
)
MANIFEST_PATH = (
    "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/"
    "supertrend_regime_manifest.json"
)
SUMMARY_PATH = "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/summary.json"
MODEL_DIR = "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
FILTER_CSV = "supertrend_filter_candidates.csv"
REPORT_MD = "supertrend_regime_audit.md"
DEFAULT_TIMEFRAMES = [5, 15]
DEFAULT_ATR_PERIODS = [5, 10, 20, 50]
WINDOW_DAYS = [5, 10, 20, 30, 50, 100, 200]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--timeframes", default=",".join(map(str, DEFAULT_TIMEFRAMES)))
    parser.add_argument("--atr-periods", default=",".join(map(str, DEFAULT_ATR_PERIODS)))
    parser.add_argument("--min-full-trades", type=int, default=100)
    parser.add_argument("--min-janmay-trades", type=int, default=30)
    return parser.parse_args()


def _rma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    out[period - 1] = np.nanmean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return _rma(tr, period)


def supertrend(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    factor: float,
    atr_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TradingView-compatible SuperTrend.

    Direction convention matches the live MGC implementation:
    -1 = bullish/up regime, +1 = bearish/down regime.
    """
    n = len(c)
    atr_val = _atr(h, l, c, atr_period)
    hl2 = (h + l) / 2.0
    upper = hl2 + factor * atr_val
    lower = hl2 - factor * atr_val
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(atr_val[i]):
            continue
        if i == 0 or np.isnan(atr_val[i - 1]):
            direction[i] = 1
            st[i] = upper[i]
            continue

        prev_lower = lower[i - 1]
        prev_upper = upper[i - 1]

        if not (lower[i] > prev_lower or c[i - 1] < prev_lower):
            lower[i] = prev_lower
        if not (upper[i] < prev_upper or c[i - 1] > prev_upper):
            upper[i] = prev_upper

        prev_st = st[i - 1]
        if np.isnan(prev_st) or np.isclose(prev_st, prev_upper, equal_nan=False):
            direction[i] = -1 if c[i] > upper[i] else 1
        else:
            direction[i] = 1 if c[i] < lower[i] else -1

        st[i] = lower[i] if direction[i] == -1 else upper[i]

    return st, direction, atr_val


def max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    equity = pd.concat([pd.Series([0.0]), pnls.reset_index(drop=True).cumsum()], ignore_index=True)
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def profit_factor(pnls: pd.Series) -> float | None:
    wins = float(pnls[pnls > 0].sum())
    losses = abs(float(pnls[pnls < 0].sum()))
    return safe_ratio(wins, losses)


def annualized_sharpe(daily_pnl: pd.Series) -> float | None:
    if len(daily_pnl) < 2:
        return None
    std = float(daily_pnl.std(ddof=1))
    if std == 0:
        return None
    return float((daily_pnl.mean() / std) * sqrt(252.0))


def annualized_sortino(daily_pnl: pd.Series) -> float | None:
    if daily_pnl.empty:
        return None
    downside = daily_pnl.clip(upper=0.0)
    downside_dev = float((downside.pow(2).mean()) ** 0.5)
    if downside_dev == 0:
        return None
    return float((daily_pnl.mean() / downside_dev) * sqrt(252.0))


def load_context(path: Path) -> pd.DataFrame:
    cols = [
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bar_data_quality_ok",
    ]
    df = pd.read_parquet(path, columns=cols).sort_values("timestamp_utc")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    numeric = ["open", "high", "low", "close", "volume"]
    df[numeric] = df[numeric].astype(float)
    df["bar_data_quality_ok"] = df["bar_data_quality_ok"].astype(bool)
    return df


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("signal_ts")
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def load_anchor(events: pd.DataFrame) -> pd.Timestamp:
    summary_path = project_path(SUMMARY_PATH)
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        anchor = pd.Timestamp(payload["signal_range"]["anchor_ts"])
        if anchor.tzinfo is None:
            return anchor.tz_localize("UTC")
        return anchor.tz_convert("UTC")
    return events["signal_ts"].max()


def build_right_labeled_bars(m1: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Build right-labeled bars from close-timestamped M1 rows.

    L1 timestamps are already shifted to the M1 close time. A 5m bar ending
    09:35 contains M1 closes 09:31, 09:32, 09:33, 09:34, 09:35, so ceil()
    gives an explicit no-lookahead close label.
    """
    rule = f"{timeframe_minutes}min"
    work = m1[m1["bar_data_quality_ok"]].copy()
    work["feature_ts"] = work["timestamp_utc"].dt.ceil(rule)
    bars = (
        work.groupby("feature_ts", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source_bar_count=("timestamp_utc", "count"),
        )
        .reset_index()
    )
    bars = bars[bars["source_bar_count"] == timeframe_minutes].copy()
    return bars.reset_index(drop=True)


def bars_since_flip(direction: np.ndarray) -> np.ndarray:
    out = np.full(len(direction), np.nan)
    last_flip: int | None = None
    prev = 0
    for i, value in enumerate(direction):
        if value == 0:
            continue
        if prev == 0 or value != prev:
            last_flip = i
        prev = int(value)
        if last_flip is not None:
            out[i] = i - last_flip
    return out


def build_st_table(
    m1: pd.DataFrame,
    timeframe_minutes: int,
    atr_periods: list[int],
    factor: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bars = build_right_labeled_bars(m1, timeframe_minutes)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)

    out = bars[["feature_ts", "close", "source_bar_count"]].copy()
    out["feature_ts"] = pd.to_datetime(out["feature_ts"], utc=True).astype("datetime64[ns, UTC]")
    out = out.rename(
        columns={
            "feature_ts": f"st{timeframe_minutes}_feature_ts",
            "close": f"st{timeframe_minutes}_bar_close",
            "source_bar_count": f"st{timeframe_minutes}_source_bar_count",
        }
    )

    valid_counts = {}
    for period in atr_periods:
        st, direction, atr_val = supertrend(h, l, c, factor=factor, atr_period=period)
        prefix = f"st{timeframe_minutes}_{period}"
        out[f"{prefix}_value"] = st
        out[f"{prefix}_dir"] = direction
        out[f"{prefix}_atr"] = atr_val
        out[f"{prefix}_close_dist_pts"] = c - st
        out[f"{prefix}_close_dist_atr"] = np.divide(
            c - st,
            atr_val,
            out=np.full(len(c), np.nan),
            where=(atr_val != 0) & ~np.isnan(atr_val),
        )
        out[f"{prefix}_flip_bars_ago"] = bars_since_flip(direction)
        valid_counts[prefix] = int(np.isfinite(st).sum())

    audit = {
        "timeframe_minutes": timeframe_minutes,
        "bars": int(len(bars)),
        "min_feature_ts": out[f"st{timeframe_minutes}_feature_ts"].min().isoformat(),
        "max_feature_ts": out[f"st{timeframe_minutes}_feature_ts"].max().isoformat(),
        "factor": factor,
        "atr_periods": atr_periods,
        "valid_st_counts": valid_counts,
    }
    return out, audit


def attach_features(
    events: pd.DataFrame,
    feature_tables: dict[int, pd.DataFrame],
    atr_periods: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = events.sort_values("signal_ts").copy()
    audit: dict[str, Any] = {"timeframes": {}, "lookahead_violations": {}}

    for timeframe, table in feature_tables.items():
        table = table.sort_values(f"st{timeframe}_feature_ts")
        out = pd.merge_asof(
            out,
            table,
            left_on="signal_ts",
            right_on=f"st{timeframe}_feature_ts",
            direction="backward",
        )
        lag_minutes = (
            (out["signal_ts"] - out[f"st{timeframe}_feature_ts"]).dt.total_seconds() / 60.0
        )
        audit["timeframes"][f"{timeframe}m"] = {
            "events_with_feature": int(out[f"st{timeframe}_feature_ts"].notna().sum()),
            "min_lag_minutes": float(lag_minutes.min()),
            "max_lag_minutes": float(lag_minutes.max()),
            "median_lag_minutes": float(lag_minutes.median()),
        }
        audit["lookahead_violations"][f"{timeframe}m"] = int((lag_minutes < 0).sum())

    for timeframe in feature_tables:
        for period in atr_periods:
            prefix = f"st{timeframe}_{period}"
            out[f"{prefix}_bullish"] = out[f"{prefix}_dir"].eq(-1)
            out[f"{prefix}_bearish"] = out[f"{prefix}_dir"].eq(1)
            out[f"{prefix}_entry_dist_pts"] = out["entry_price"].astype(float) - out[f"{prefix}_value"]
            out[f"{prefix}_entry_dist_to_risk"] = np.divide(
                out[f"{prefix}_entry_dist_pts"],
                out["entry_risk_pts"].astype(float),
                out=np.full(len(out), np.nan),
                where=out["entry_risk_pts"].astype(float).to_numpy() != 0,
            )

    return out, audit


def summarize_subset(events: pd.DataFrame, prefix: str) -> dict[str, Any]:
    pnl = events["pnl_net_usd"].astype(float)
    return {
        f"{prefix}_trades": int(len(events)),
        f"{prefix}_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
        f"{prefix}_max_dd": max_drawdown(pnl) if not pnl.empty else 0.0,
        f"{prefix}_win_rate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
        f"{prefix}_profit_factor": profit_factor(pnl) if not pnl.empty else None,
        f"{prefix}_avg_trade": float(pnl.mean()) if not pnl.empty else 0.0,
    }


def daily_metrics(events: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if events.empty:
        return {
            f"{prefix}_daily_sharpe": None,
            f"{prefix}_daily_sortino": None,
            f"{prefix}_best_day": 0.0,
            f"{prefix}_worst_day": 0.0,
        }
    daily = (
        events.assign(_date=pd.to_datetime(events["ny_date"]))
        .groupby("_date")["pnl_net_usd"]
        .sum()
        .astype(float)
    )
    return {
        f"{prefix}_daily_sharpe": annualized_sharpe(daily),
        f"{prefix}_daily_sortino": annualized_sortino(daily),
        f"{prefix}_best_day": float(daily.max()),
        f"{prefix}_worst_day": float(daily.min()),
    }


def candidate_summary(
    events: pd.DataFrame,
    feature_names: list[str],
    mask: pd.Series,
    name: str,
    anchor: pd.Timestamp,
) -> dict[str, Any]:
    selected = events[mask.fillna(False)].copy()
    dates = pd.to_datetime(selected["ny_date"])
    janmay = selected[(dates >= pd.Timestamp("2026-01-01")) & (dates <= pd.Timestamp("2026-05-31"))]
    march = selected[(dates >= pd.Timestamp("2026-03-01")) & (dates <= pd.Timestamp("2026-03-31"))]
    aprmay = selected[(dates >= pd.Timestamp("2026-04-01")) & (dates <= pd.Timestamp("2026-05-31"))]

    row: dict[str, Any] = {
        "candidate": name,
        "filter_count": int(len(feature_names)),
        "filters": ",".join(feature_names) if feature_names else "BASELINE",
    }
    row.update(summarize_subset(selected, "full"))
    row.update(daily_metrics(selected, "full"))
    row.update(summarize_subset(janmay, "jan_may_2026"))
    row.update(summarize_subset(march, "march_2026"))
    row.update(summarize_subset(aprmay, "apr_may_2026"))
    for days in WINDOW_DAYS:
        window = selected[
            (selected["signal_ts"] > anchor - pd.Timedelta(days=days))
            & (selected["signal_ts"] <= anchor)
        ]
        row.update(summarize_subset(window, f"last_{days}d"))
    return row


def evaluate_candidates(events: pd.DataFrame, feature_names: list[str], anchor: pd.Timestamp) -> pd.DataFrame:
    rows = [
        candidate_summary(
            events,
            [],
            pd.Series(True, index=events.index),
            "BASELINE",
            anchor,
        )
    ]
    for size in range(1, len(feature_names) + 1):
        for combo in combinations(feature_names, size):
            mask = pd.Series(True, index=events.index)
            for feature in combo:
                mask &= events[f"{feature.lower()}_bullish"].fillna(False)
            rows.append(
                candidate_summary(
                    events,
                    list(combo),
                    mask,
                    " & ".join(combo),
                    anchor,
                )
            )
    out = pd.DataFrame(rows)
    baseline = out[out["candidate"] == "BASELINE"].iloc[0]
    out["full_trade_retention"] = out["full_trades"] / baseline["full_trades"]
    out["jan_may_trade_retention"] = out["jan_may_2026_trades"] / baseline["jan_may_2026_trades"]
    if baseline["apr_may_2026_pnl"] != 0:
        out["apr_may_pnl_retention"] = out["apr_may_2026_pnl"] / baseline["apr_may_2026_pnl"]
    else:
        out["apr_may_pnl_retention"] = np.nan
    out["march_dd_improvement"] = out["march_2026_max_dd"] - baseline["march_2026_max_dd"]
    out["march_pnl_improvement"] = out["march_2026_pnl"] - baseline["march_2026_pnl"]
    out["full_return_dd"] = np.divide(
        out["full_pnl"],
        out["full_max_dd"].abs(),
        out=np.full(len(out), np.nan),
        where=out["full_max_dd"].abs() != 0,
    )
    return out.sort_values(["filter_count", "candidate"]).reset_index(drop=True)


def fmt_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${float(value):,.0f}"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1%}"


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str]], limit: int = 15) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i, _ in enumerate(columns)) + " |",
    ]
    for _, row in df.head(limit).iterrows():
        values = []
        for col, _ in columns:
            value = row[col]
            if col.endswith("_pnl") or col.endswith("_max_dd") or col in {
                "march_dd_improvement",
                "march_pnl_improvement",
            }:
                values.append(fmt_money(value))
            elif col.endswith("_win_rate") or col.endswith("_retention"):
                values.append(fmt_pct(value))
            elif "profit_factor" in col or col == "full_return_dd":
                values.append(fmt_num(value))
            elif col.endswith("_trades") or col == "filter_count":
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    candidates: pd.DataFrame,
    manifest: dict[str, Any],
    min_full_trades: int,
    min_janmay_trades: int,
) -> None:
    baseline = candidates[candidates["candidate"] == "BASELINE"].iloc[0]
    liquid = candidates[
        (candidates["candidate"] != "BASELINE")
        & (candidates["full_trades"] >= min_full_trades)
        & (candidates["jan_may_2026_trades"] >= min_janmay_trades)
    ].copy()

    march_rank = liquid.sort_values(
        ["march_dd_improvement", "jan_may_2026_pnl", "full_return_dd"],
        ascending=[False, False, False],
    )
    full_rank = liquid.sort_values(
        ["full_return_dd", "full_pnl", "jan_may_2026_pnl"],
        ascending=[False, False, False],
    )
    recent_rank = liquid.sort_values(
        ["last_30d_pnl", "last_30d_max_dd", "jan_may_2026_pnl"],
        ascending=[False, False, False],
    )

    summary_cols = [
        ("candidate", "Candidate"),
        ("filter_count", "N"),
        ("full_trades", "Full Trades"),
        ("full_pnl", "Full PnL"),
        ("full_max_dd", "Full DD"),
        ("full_return_dd", "Ret/DD"),
        ("jan_may_2026_trades", "Jan-May Trades"),
        ("jan_may_2026_pnl", "Jan-May PnL"),
        ("jan_may_2026_max_dd", "Jan-May DD"),
        ("march_2026_pnl", "Mar PnL"),
        ("march_2026_max_dd", "Mar DD"),
        ("march_dd_improvement", "Mar DD Improve"),
        ("last_30d_pnl", "30D PnL"),
        ("last_30d_max_dd", "30D DD"),
    ]

    lines = [
        "# MNQ ORB SuperTrend Regime Audit",
        "",
        f"Strategy baseline: `{STRATEGY_ID}`",
        "",
        "## Scope",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Source events | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet` |",
        "| Feature source | `data/Level_1_Features/mnq/ORB/context.parquet` |",
        "| SuperTrend factor | " + fmt_num(manifest["params"]["factor"]) + " |",
        "| Timeframes | " + ", ".join(f"{x}m" for x in manifest["params"]["timeframes"]) + " |",
        "| ATR periods | " + ", ".join(map(str, manifest["params"]["atr_periods"])) + " |",
        "| Direction convention | `-1 = bullish/up`, `+1 = bearish/down` |",
        "| Join rule | Latest completed ST feature timestamp `<= signal_ts`; entry remains next M1 open |",
        "| Candidate grid | All non-empty bullish conjunctions across 8 ST states, plus baseline |",
        "",
        "## Baseline Reference",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Full trades | {int(baseline['full_trades']):,} |",
        f"| Full PnL | {fmt_money(baseline['full_pnl'])} |",
        f"| Full max DD | {fmt_money(baseline['full_max_dd'])} |",
        f"| Full Ret/DD | {fmt_num(baseline['full_return_dd'])} |",
        f"| Jan-May 2026 trades | {int(baseline['jan_may_2026_trades']):,} |",
        f"| Jan-May 2026 PnL | {fmt_money(baseline['jan_may_2026_pnl'])} |",
        f"| Jan-May 2026 max DD | {fmt_money(baseline['jan_may_2026_max_dd'])} |",
        f"| March 2026 PnL | {fmt_money(baseline['march_2026_pnl'])} |",
        f"| March 2026 max DD | {fmt_money(baseline['march_2026_max_dd'])} |",
        f"| Last 30D PnL | {fmt_money(baseline['last_30d_pnl'])} |",
        f"| Last 30D max DD | {fmt_money(baseline['last_30d_max_dd'])} |",
        "",
        "## Top Filters For March Drawdown",
        "",
        "Filtered to candidates with enough sample: "
        f"`full_trades >= {min_full_trades}` and `jan_may_2026_trades >= {min_janmay_trades}`.",
        "",
    ]
    lines.extend(markdown_table(march_rank, summary_cols, limit=15))
    lines.extend(
        [
            "",
            "## Top Filters By Full Ret/DD",
            "",
        ]
    )
    lines.extend(markdown_table(full_rank, summary_cols, limit=15))
    lines.extend(
        [
            "",
            "## Top Filters By Last 30D PnL",
            "",
        ]
    )
    lines.extend(markdown_table(recent_rank, summary_cols, limit=15))
    lines.extend(
        [
            "",
            "## Feature Audit",
            "",
            "| Check | Value |",
            "| --- | ---: |",
            f"| Event rows enriched | {manifest['events']['rows']:,} |",
            f"| Feature columns added | {manifest['features']['feature_columns_added']:,} |",
            f"| Candidate rows | {manifest['candidates']['rows']:,} |",
            f"| Lookahead violations | {manifest['lookahead']['total_violations']:,} |",
            f"| Max feature lag minutes | {fmt_num(manifest['lookahead']['max_lag_minutes'])} |",
            "",
            "## Artifacts",
            "",
            "| Artifact | Path |",
            "| --- | --- |",
            f"| Enriched event features | `{manifest['artifacts']['features_path']}` |",
            f"| Candidate CSV | `{manifest['artifacts']['candidate_csv']}` |",
            f"| Manifest | `{manifest['artifacts']['manifest_path']}` |",
            "",
            "## Current Read",
            "",
            "- This is an audit/filter layer, not a new live rule.",
            "- A filter is useful only if it improves March-like drawdown without destroying April-May continuation.",
            "- Any selected candidate still needs walk-forward validation before promotion.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg = load_config()
    assert_mnq_namespaces(cfg)
    timeframes = parse_int_list(args.timeframes)
    atr_periods = parse_int_list(args.atr_periods)

    features_path = project_path(FEATURES_PATH)
    manifest_path = project_path(MANIFEST_PATH)
    model_dir = project_path(MODEL_DIR)
    candidate_csv = model_dir / FILTER_CSV
    report_path = model_dir / REPORT_MD
    for path in [features_path, manifest_path, candidate_csv, report_path]:
        if path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing artifact without --force: {path}")

    events = load_events(project_path(EVENTS_PATH))
    anchor = load_anchor(events)
    m1 = load_context(project_path(cfg["outputs"]["l1_context"]))

    feature_tables: dict[int, pd.DataFrame] = {}
    table_audits = []
    for timeframe in timeframes:
        table, table_audit = build_st_table(m1, timeframe, atr_periods, factor=args.factor)
        feature_tables[timeframe] = table
        table_audits.append(table_audit)

    enriched, attach_audit = attach_features(events, feature_tables, atr_periods)
    feature_names = [f"ST{timeframe}_{period}" for timeframe in timeframes for period in atr_periods]
    candidates = evaluate_candidates(enriched, feature_names, anchor)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(features_path, index=False)
    model_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidate_csv, index=False)

    lag_values = []
    for timeframe in timeframes:
        lag = (
            (enriched["signal_ts"] - enriched[f"st{timeframe}_feature_ts"]).dt.total_seconds()
            / 60.0
        )
        lag_values.extend(lag.dropna().tolist())
    feature_cols = [c for c in enriched.columns if c.startswith("st")]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": STRATEGY_ID,
        "params": {
            "factor": args.factor,
            "timeframes": timeframes,
            "atr_periods": atr_periods,
            "direction_convention": "-1=bullish/up, +1=bearish/down",
        },
        "events": {
            "rows": int(len(enriched)),
            "min_signal_ts": enriched["signal_ts"].min().isoformat(),
            "max_signal_ts": enriched["signal_ts"].max().isoformat(),
            "anchor_ts": anchor.isoformat(),
        },
        "features": {
            "feature_columns_added": int(len(feature_cols)),
            "feature_names": feature_names,
            "table_audits": table_audits,
        },
        "lookahead": {
            "rule": "feature_ts <= signal_ts for every attached timeframe",
            "by_timeframe": attach_audit,
            "total_violations": int(
                sum(attach_audit["lookahead_violations"].values())
            ),
            "min_lag_minutes": float(min(lag_values)) if lag_values else None,
            "max_lag_minutes": float(max(lag_values)) if lag_values else None,
        },
        "candidates": {
            "rows": int(len(candidates)),
            "grid": "all non-empty bullish conjunctions across ST feature names plus baseline",
        },
        "artifacts": {
            "features_path": FEATURES_PATH,
            "candidate_csv": str(candidate_csv.relative_to(project_path("."))),
            "report_path": str(report_path.relative_to(project_path("."))),
            "manifest_path": MANIFEST_PATH,
        },
    }
    write_json(manifest_path, manifest)
    write_report(report_path, candidates, manifest, args.min_full_trades, args.min_janmay_trades)

    print(f"Wrote {features_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {candidate_csv}")
    print(f"Wrote {report_path}")
    print(
        candidates.sort_values(
            ["march_dd_improvement", "jan_may_2026_pnl", "full_return_dd"],
            ascending=[False, False, False],
        )
        .head(10)[
            [
                "candidate",
                "full_trades",
                "full_pnl",
                "full_max_dd",
                "jan_may_2026_trades",
                "jan_may_2026_pnl",
                "march_2026_pnl",
                "march_2026_max_dd",
                "march_dd_improvement",
                "last_30d_pnl",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
