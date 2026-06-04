from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod"
MODEL_DIR = ROOT / "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
CHART_DIR = MODEL_DIR / "charts"
MONTE_DIR = MODEL_DIR / "monte_carlo"
L0_DIR = ROOT / "data/Level_0_Raw"
L1_DIR = ROOT / "data/Level_1_Features/mnq/ORB"
SWEEP_DIR = ROOT / "data/Level_2_Datamart/mnq/ORB/sweeps"
RAW_BASE = (
    "https://raw.githubusercontent.com/kemtol/FFFUTURES/main/"
    "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
)
ST_VARIANT_CSV = MODEL_DIR / "supertrend_variant_comparison.csv"
ST_FILTER_CSV = MODEL_DIR / "supertrend_filter_candidates.csv"
ST_REGIME_MANIFEST = DATA_DIR / "supertrend_regime_manifest.json"
ST_VARIANT_MANIFEST = DATA_DIR / "supertrend_variant_comparison_manifest.json"
SHORT_SWITCH_CSV = MODEL_DIR / "short_reversal_switch_comparison.csv"
SHORT_SWITCH_MANIFEST = DATA_DIR / "short_reversal_switch_comparison_manifest.json"
SHORT_SWITCH_P0_CSV = MODEL_DIR / "short_switch_tp2r_p0_sweep.csv"
SHORT_SWITCH_P0_REPORT = MODEL_DIR / "short_switch_tp2r_p0_sweep.md"
SHORT_SWITCH_P0_FULL_REPORT = MODEL_DIR / "short_switch_tp2r_p0_full_report.md"
SHORT_SWITCH_P0_BEST_EVENTS = MODEL_DIR / "short_switch_tp2r_p0_best_events.csv"
SHORT_SWITCH_P0_BEST_LEGS = MODEL_DIR / "short_switch_tp2r_p0_best_legs.csv"
SHORT_SWITCH_P0_BEST_YEARLY = MODEL_DIR / "short_switch_tp2r_p0_best_yearly.csv"
SHORT_SWITCH_P0_BEST_MONTHLY = MODEL_DIR / "short_switch_tp2r_p0_best_monthly.csv"
SHORT_SWITCH_P0_MANIFEST = DATA_DIR / "short_switch_tp2r_p0_sweep_manifest.json"
PACKAGE_GATE = DATA_DIR / "package_gate.json"
L0_1M_MANIFEST = L0_DIR / "MNQ_1m_duckdb_manifest.json"
L0_1M_YF_MANIFEST = L0_DIR / "MNQ_1m_yfinance_append_manifest.json"
L0_CONTINUITY_REPORT = L0_DIR / "MNQ_1m_continuity_report.json"
L0_5M_MANIFEST = L0_DIR / "MNQ_5m_duckdb_manifest.json"
L0_15M_MANIFEST = L0_DIR / "MNQ_15m_duckdb_manifest.json"
L0_PARITY_REPORT = L0_DIR / "MNQ_yfinance_timeframe_parity_report.json"
L0_DAILY_MANIFEST = L0_DIR / "yfinance_daily_manifest.json"
L1_CONTEXT_MANIFEST = L1_DIR / "context_manifest.json"
L1_AUDIT = L1_DIR / "l1_audit.json"
L1_DAILY_CONFLUENCE_MANIFEST = L1_DIR / "daily_confluence_manifest.json"
L1_DAILY_CONFLUENCE_AUDIT = L1_DIR / "daily_confluence_audit.json"
SWEEP_MANIFEST = SWEEP_DIR / "sweep_manifest.json"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def raw(path: str) -> str:
    return f"{RAW_BASE}/{path}"


def usd(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_inputs() -> tuple[pd.DataFrame, dict]:
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    events["ny_date"] = pd.to_datetime(events["ny_date"])
    events = events.sort_values(["ny_date", "entry_ts"]).reset_index(drop=True)

    with (DATA_DIR / "summary.json").open() as f:
        summary = json.load(f)

    return events, summary


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def clean_svg(path: Path) -> None:
    text = path.read_text()
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def save_fig(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    svg_path = directory / f"{stem}.svg"
    png_path = directory / f"{stem}.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=150)
    clean_svg(svg_path)


def save_equity_curve(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    equity = events["pnl_net_usd"].cumsum()
    ax.plot(events["ny_date"], equity, color="#0f766e", linewidth=1.8)
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.5)
    style_axis(ax, "NASDAQ Micro Futures ORB Rule-Based Equity Curve", "Cumulative PnL ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "equity_curve")
    plt.close(fig)


def save_drawdown_curve(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    equity = events["pnl_net_usd"].cumsum()
    drawdown = equity - equity.cummax()
    ax.fill_between(events["ny_date"], drawdown, 0, color="#dc2626", alpha=0.28)
    ax.plot(events["ny_date"], drawdown, color="#991b1b", linewidth=1.2)
    style_axis(ax, "NASDAQ Micro Futures ORB Rule-Based Drawdown", "Drawdown ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "drawdown_curve")
    plt.close(fig)


def save_monthly_pnl(events: pd.DataFrame) -> None:
    monthly = events.assign(month=events["ny_date"].dt.to_period("M").astype(str))
    monthly = monthly.groupby("month", as_index=False)["pnl_net_usd"].sum()

    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = ["#0f766e" if x >= 0 else "#dc2626" for x in monthly["pnl_net_usd"]]
    ax.bar(monthly["month"], monthly["pnl_net_usd"], color=colors, width=0.85)
    ax.axhline(0, color="#334155", linewidth=0.8)
    style_axis(ax, "Monthly Net PnL", "PnL ($)")
    tick_step = max(1, len(monthly) // 14)
    ax.set_xticks(range(0, len(monthly), tick_step))
    ax.set_xticklabels(monthly["month"].iloc[::tick_step], rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "monthly_pnl")
    plt.close(fig)


def save_rolling_windows(summary: dict) -> None:
    order = ["5D", "10D", "20D", "30D", "50D", "100D", "200D"]
    rows = [
        {
            "window": window,
            "pnl": summary["windows"][window]["pnl_usd"],
            "dd": summary["windows"][window]["max_dd_usd"],
        }
        for window in order
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(df))
    ax.bar([i - 0.18 for i in x], df["pnl"], width=0.36, color="#0f766e", label="PnL")
    ax.bar([i + 0.18 for i in x], df["dd"], width=0.36, color="#dc2626", label="Max DD")
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["window"])
    ax.legend(frameon=False)
    style_axis(ax, "Recent Rolling Window PnL / DD", "$")
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "rolling_windows")
    plt.close(fig)


def save_trade_pnl_distribution(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(events["pnl_net_usd"], bins=45, color="#2563eb", alpha=0.78)
    ax.axvline(0, color="#334155", linewidth=0.9)
    style_axis(ax, "Trade PnL Distribution", "Trade count")
    ax.set_xlabel("Net PnL per trade ($)")
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "trade_pnl_distribution")
    plt.close(fig)


def build_daily_pnl(events: pd.DataFrame) -> pd.Series:
    daily = events.groupby(events["ny_date"].dt.normalize())["pnl_net_usd"].sum().sort_index()
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    return daily.reindex(idx, fill_value=0.0)


def monte_carlo(daily_pnl: pd.Series, n_paths: int = 5000, seed: int = 260531) -> dict:
    rng = np.random.default_rng(seed)
    values = daily_pnl.to_numpy(dtype=float)
    horizons = [30, 100, 200]
    result: dict[str, dict] = {}

    for horizon in horizons:
        samples = rng.choice(values, size=(n_paths, horizon), replace=True)
        cumulative = samples.cumsum(axis=1)
        with_zero = np.concatenate([np.zeros((n_paths, 1)), cumulative], axis=1)
        peaks = np.maximum.accumulate(with_zero, axis=1)[:, 1:]
        drawdowns = cumulative - peaks
        final_pnl = cumulative[:, -1]
        max_dd = drawdowns.min(axis=1)

        result[f"{horizon}D"] = {
            "horizon": horizon,
            "paths": n_paths,
            "median_pnl_usd": float(np.median(final_pnl)),
            "p5_pnl_usd": float(np.percentile(final_pnl, 5)),
            "p95_pnl_usd": float(np.percentile(final_pnl, 95)),
            "prob_final_loss": float((final_pnl < 0).mean()),
            "median_max_dd_usd": float(np.median(max_dd)),
            "p5_max_dd_usd": float(np.percentile(max_dd, 5)),
            "prob_dd_breach_2000": float((max_dd <= -2000).mean()),
            "prob_hit_3000": float((cumulative.max(axis=1) >= 3000).mean()),
            "final_pnl": final_pnl,
            "max_dd": max_dd,
            "sample_paths": cumulative[:250],
        }

    return result


def save_monte_carlo_charts(mc: dict) -> None:
    for key in ["30D", "100D"]:
        horizon = mc[key]["horizon"]
        paths = mc[key]["sample_paths"]
        days = np.arange(1, horizon + 1)

        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.plot(days, paths.T, color="#64748b", alpha=0.05, linewidth=0.8)
        ax.plot(days, np.median(paths, axis=0), color="#0f766e", linewidth=2.0, label="Median")
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.axhline(3000, color="#2563eb", linewidth=0.9, linestyle="--", label="+$3,000")
        style_axis(ax, f"Monte Carlo PnL Fan {key}", "Cumulative PnL ($)")
        ax.set_xlabel("Trading days")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_fig(fig, MONTE_DIR, f"monte_pnl_fan_{key.lower()}")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    final_pnl = np.sort(mc["30D"]["final_pnl"])
    cdf = np.arange(1, len(final_pnl) + 1) / len(final_pnl)
    ax.plot(final_pnl, cdf, color="#2563eb", linewidth=1.8)
    ax.axvline(0, color="#334155", linewidth=0.9)
    ax.axvline(3000, color="#0f766e", linewidth=0.9, linestyle="--")
    style_axis(ax, "Monte Carlo Final PnL CDF 30D", "Cumulative probability")
    ax.set_xlabel("Final PnL ($)")
    fig.tight_layout()
    save_fig(fig, MONTE_DIR, "monte_final_pnl_cdf_30d")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(mc["30D"]["max_dd"], bins=55, color="#dc2626", alpha=0.78)
    ax.axvline(-2000, color="#334155", linewidth=1.0, linestyle="--", label="-$2,000")
    style_axis(ax, "Monte Carlo Max Drawdown 30D", "Path count")
    ax.set_xlabel("Max drawdown ($)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, MONTE_DIR, "monte_maxdd_hist_30d")
    plt.close(fig)


def monte_rows(mc: dict) -> str:
    rows = []
    for key in ["30D", "100D", "200D"]:
        item = mc[key]
        rows.append(
            "| {key} | {median} | {p5} | {loss} | {dd} | {mll} | {target} |".format(
                key=key,
                median=usd(item["median_pnl_usd"]),
                p5=usd(item["p5_pnl_usd"]),
                loss=pct(item["prob_final_loss"]),
                dd=usd(item["median_max_dd_usd"]),
                mll=pct(item["prob_dd_breach_2000"]),
                target=pct(item["prob_hit_3000"]),
            )
        )
    return "\n".join(rows)


def last_trades_rows(events: pd.DataFrame) -> str:
    rows = []
    for _, row in events.tail(10).iterrows():
        rows.append(
            "| {date} | {signal} | {exit_reason} | {contracts} | {pnl} |".format(
                date=row["ny_date"].strftime("%Y-%m-%d"),
                signal=str(row["signal_ts"])[:16],
                exit_reason=row["exit_reason"],
                contracts=int(row["contracts_used"]),
                pnl=usd(float(row["pnl_net_usd"])),
            )
        )
    return "\n".join(rows)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def maybe_num(value) -> str:
    parsed = optional_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.2f}"


def pass_fail(value: bool | None) -> str:
    if value is None:
        return "UNKNOWN"
    return "PASS" if value else "FAIL"


def zero_pass(value) -> str:
    parsed = optional_float(value)
    if parsed is None:
        return "UNKNOWN"
    return "PASS" if parsed == 0 else "FAIL"


def data_lineage_control_rows(events: pd.DataFrame) -> str:
    l0_continuity = safe_read_json(L0_CONTINUITY_REPORT) or {}
    l0_yf = safe_read_json(L0_1M_YF_MANIFEST) or {}
    l0_5m = safe_read_json(L0_5M_MANIFEST) or {}
    l0_15m = safe_read_json(L0_15M_MANIFEST) or {}
    l0_parity = safe_read_json(L0_PARITY_REPORT) or {}
    l1_audit = safe_read_json(L1_AUDIT) or {}
    l1_context_manifest = safe_read_json(L1_CONTEXT_MANIFEST) or {}
    l1_daily = safe_read_json(L1_DAILY_CONFLUENCE_MANIFEST) or {}
    l1_daily_audit = safe_read_json(L1_DAILY_CONFLUENCE_AUDIT) or {}
    st_manifest = safe_read_json(ST_REGIME_MANIFEST) or {}
    st_variant_manifest = safe_read_json(ST_VARIANT_MANIFEST) or {}
    package_gate = safe_read_json(PACKAGE_GATE) or {}

    event_required = [
        "ny_date",
        "signal_ts",
        "entry_ts",
        "exit_ts",
        "side",
        "orb_high",
        "orb_low",
        "entry_price",
        "exit_price",
        "contracts_used",
        "pnl_net_usd",
    ]
    available_required = [c for c in event_required if c in events.columns]
    event_nulls = int(events[available_required].isna().sum().sum()) if available_required else -1
    event_duplicates = int(events["ny_date"].duplicated().sum()) if "ny_date" in events.columns else -1
    entry_timing_bad = 0
    exit_timing_bad = 0
    if {"signal_ts", "entry_ts", "exit_ts"}.issubset(events.columns):
        signal_ts = pd.to_datetime(events["signal_ts"], utc=True)
        entry_ts = pd.to_datetime(events["entry_ts"], utc=True)
        exit_ts = pd.to_datetime(events["exit_ts"], utc=True)
        entry_timing_bad = int((entry_ts <= signal_ts).sum())
        exit_timing_bad = int((exit_ts < entry_ts).sum())

    l0_base = l0_continuity.get("base", {})
    gap_summary = l0_continuity.get("gap_summary", {})
    l0_integrity_status = pass_fail(l0_continuity.get("hard_integrity_pass"))
    l1_status = l1_audit.get("status", "UNKNOWN")
    daily_status = l1_daily_audit.get("status") or l1_daily.get("status", "UNKNOWN")
    parity_status = l0_parity.get("status", "UNKNOWN")
    st_lookahead = int((st_manifest.get("lookahead") or {}).get("total_violations", 0))
    st_variant_lookahead = int(st_variant_manifest.get("lookahead_violations", 0))
    daily_lookahead = int(l1_daily.get("lookahead_violations", 0))

    rows = [
        "| Bronze / L0 M1 OHLCV | `data/Level_0_Raw/MNQ_1m.duckdb` | {rows:,} rows, {min_ts} to {max_ts}; hard integrity {status}; duplicates {dupes}; null OHLCV {nulls}; bad OHLC {bad_ohlc}; negative volume {neg_vol} |".format(
            rows=int(l0_base.get("rows", 0)),
            min_ts=l0_base.get("min_ts", "n/a"),
            max_ts=l0_base.get("max_ts", "n/a"),
            status=l0_integrity_status,
            dupes=int(l0_continuity.get("duplicate_timestamps", 0)),
            nulls=int(l0_base.get("null_ohlcv", 0)),
            bad_ohlc=int(l0_base.get("bad_high_rows", 0)) + int(l0_base.get("bad_low_rows", 0)),
            neg_vol=int(l0_base.get("negative_volume_rows", 0)),
        ),
        "| L0 continuity | `data/Level_0_Raw/MNQ_1m_continuity_report.json` | {status}; gaps >60s {gaps:,}; max gap {max_gap:,}s; downstream rule: quarantine gap bars, do not train across gaps |".format(
            status=l0_continuity.get("continuity_status", "UNKNOWN"),
            gaps=int(gap_summary.get("gap_count_gt_60s", 0)),
            max_gap=int(gap_summary.get("max_gap_seconds", 0)),
        ),
        "| Recent yfinance append | `data/Level_0_Raw/MNQ_1m_yfinance_append_manifest.json` | {rows:,} appended/replaced rows, {first_ts} to {last_ts}; post-append duplicates {dupes} |".format(
            rows=int((l0_yf.get("append") or {}).get("rows_inserted", 0)),
            first_ts=(l0_yf.get("append") or {}).get("first_ts", "n/a"),
            last_ts=(l0_yf.get("append") or {}).get("last_ts", "n/a"),
            dupes=int((l0_yf.get("after") or {}).get("duplicate_timestamps", 0)),
        ),
        "| Derived L0 5m/15m | `MNQ_5m.duckdb`, `MNQ_15m.duckdb` | right-labeled, left-closed from M1; 5m rows {rows_5m:,}, 15m rows {rows_15m:,}; duplicate timestamps 5m/15m = {dup_5m}/{dup_15m} |".format(
            rows_5m=int(l0_5m.get("rows", 0)),
            rows_15m=int(l0_15m.get("rows", 0)),
            dup_5m=int(l0_5m.get("duplicate_timestamps", 0)),
            dup_15m=int(l0_15m.get("duplicate_timestamps", 0)),
        ),
        "| yfinance timeframe parity | `data/Level_0_Raw/MNQ_yfinance_timeframe_parity_report.json` | {status}; max mismatch rate {rate:.2%}; latest incomplete bars excluded {excluded} |".format(
            status=parity_status,
            rate=float(l0_parity.get("max_mismatch_rate", 0.0)),
            excluded=int(l0_parity.get("excluded_latest_bars", 0)),
        ),
        "| L1 intraday context | `data/Level_1_Features/mnq/ORB/context.parquet` | {status}; {rows:,} rows, {cols} columns; {days:,} NY days; OR-complete days {orb_days:,}; duplicate timestamps {dupes}; bad OHLC rows {bad_ohlc}; required hard nulls {hard_nulls} |".format(
            status=l1_status,
            rows=int(l1_audit.get("rows", l1_context_manifest.get("rows", 0))),
            cols=int(l1_audit.get("columns", 0)),
            days=int(l1_audit.get("ny_days", l1_context_manifest.get("ny_days", 0))),
            orb_days=int(l1_audit.get("orb_complete_days", l1_context_manifest.get("orb_complete_days", 0))),
            dupes=int(l1_audit.get("duplicate_timestamps", 0)),
            bad_ohlc=int(l1_audit.get("bad_ohlc_rows", 0)),
            hard_nulls=sum(int(v) for v in (l1_audit.get("hard_nulls") or {}).values()),
        ),
        "| L1 ORB leakage guards | `data/Level_1_Features/mnq/ORB/l1_audit.json` | pre-OR rows with OR values {pre_or}; eligible rows before OR end {eligible_before}; eligible bad-quality rows {eligible_bad}; complete OR days wrong bar count {wrong_count} |".format(
            pre_or=int(l1_audit.get("pre_or_rows_with_or_values", 0)),
            eligible_before=int(l1_audit.get("eligible_rows_before_or_end", 0)),
            eligible_bad=int(l1_audit.get("eligible_rows_bad_quality", 0)),
            wrong_count=int(l1_audit.get("complete_orb_days_wrong_bar_count", 0)),
        ),
        "| Daily confluence features | `data/Level_1_Features/mnq/ORB/daily_confluence.parquet` | {status}; {rows:,} rows, {features} features; feature nulls {feature_nulls}; lookahead violations {lookahead}; contract: external daily feature date < MNQ trade date |".format(
            status=daily_status,
            rows=int(l1_daily.get("rows", 0)),
            features=int(l1_daily.get("feature_count", 0)),
            feature_nulls=sum(int(v) for v in (l1_daily.get("feature_nulls") or {}).values()),
            lookahead=daily_lookahead,
        ),
        "| L2 baseline events | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet` | {rows:,} rows, {cols} columns; required nulls {nulls}; duplicate NY dates {dupes}; entry<=signal rows {bad_entry}; exit<entry rows {bad_exit} |".format(
            rows=len(events),
            cols=len(events.columns),
            nulls=event_nulls,
            dupes=event_duplicates,
            bad_entry=entry_timing_bad,
            bad_exit=exit_timing_bad,
        ),
        "| Frozen package gate | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/package_gate.json` | {status}; events {rows:,}; failures {failures}; warnings {warnings} |".format(
            status=package_gate.get("status", "UNKNOWN"),
            rows=int(package_gate.get("events_rows", 0)),
            failures=len(package_gate.get("failures") or {}),
            warnings=len(package_gate.get("warnings") or {}),
        ),
        "| SuperTrend feature attach | `supertrend_regime_features.parquet` | lookahead violations {lookahead}; variant attach lookahead {variant_lookahead}; max feature lag {max_lag} minutes |".format(
            lookahead=st_lookahead,
            variant_lookahead=st_variant_lookahead,
            max_lag=(st_manifest.get("lookahead") or {}).get("max_lag_minutes", "n/a"),
        ),
    ]
    return "\n".join(rows)


def data_feature_dataset_rows(events: pd.DataFrame) -> str:
    l1_audit = safe_read_json(L1_AUDIT) or {}
    l1_daily = safe_read_json(L1_DAILY_CONFLUENCE_MANIFEST) or {}
    st_manifest = safe_read_json(ST_REGIME_MANIFEST) or {}
    sweep_manifest = safe_read_json(SWEEP_MANIFEST) or {}
    l0_continuity = safe_read_json(L0_CONTINUITY_REPORT) or {}
    l0_base = l0_continuity.get("base", {})

    st_cols = int((st_manifest.get("features") or {}).get("feature_columns_added", 0))
    return "\n".join(
        [
            "| Bronze/L0 raw | `MNQ_1m.duckdb` | OHLCV M1, source symbol, volume | {rows:,} rows | Raw market bars; if this layer is wrong, all reports are invalid |".format(
                rows=int(l0_base.get("rows", 0))
            ),
            "| L1 intraday context | `context.parquet` | OHLCV, NY date/time, `minutes_from_open`, quality flags, OR context | {rows:,} rows / {cols} cols | Research context; sweep recomputes 10/15/20/30m OR only from quality bars |".format(
                rows=int(l1_audit.get("rows", 0)),
                cols=int(l1_audit.get("columns", 0)),
            ),
            "| L1 daily confluence | `daily_confluence.parquet` | SPY, QQQ, VIX, TNX, DXY prior-day features | {rows:,} rows / {features} features | Not used by this rule-based report, but available for ML overlays with strict D-1 contract |".format(
                rows=int(l1_daily.get("rows", 0)),
                features=int(l1_daily.get("feature_count", 0)),
            ),
            "| L2 sweep | `sweeps/sweep_events.parquet` | ORB parameter grid events | {rows:,} rows | Parent grid for selecting current baseline candidate |".format(
                rows=int(sweep_manifest.get("event_rows", 0))
            ),
            "| L2 baseline package | `rule_based_15m_long_tp2r_eod/events.parquet` | Executed trade events, sizing, cost-adjusted PnL | {rows:,} rows / {cols} cols | Current benchmark/control strategy |".format(
                rows=len(events),
                cols=len(events.columns),
            ),
            "| L2 ST feature package | `supertrend_regime_features.parquet` | ST5/ST15 ATR 5/10/20/50 values, dirs, distances, lags | {rows:,} rows / {cols} feature cols added | Regime-filter audit only; attached with feature timestamp <= signal timestamp |".format(
                rows=int((st_manifest.get("events") or {}).get("rows", len(events))),
                cols=st_cols,
            ),
        ]
    )


def variant_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "| {label} | {trades:,} | {longs:,} | {shorts:,} | {wr} | {pnl} | {dd} | {retdd} | {jm_trades:,} | {jm_pnl} | {jm_dd} | {mar_pnl} | {mar_dd} | {d30_trades:,} | {d30_pnl} | {d30_dd} |".format(
                label=row["label"],
                trades=int(row["trades"]),
                longs=int(row["long_trades"]),
                shorts=int(row["short_trades"]),
                wr=pct(float(row["win_rate"])),
                pnl=usd(float(row["pnl_usd"])),
                dd=usd(float(row["max_dd_usd"])),
                retdd=maybe_num(row["return_dd"]),
                jm_trades=int(row["jan_may_2026_trades"]),
                jm_pnl=usd(float(row["jan_may_2026_pnl_usd"])),
                jm_dd=usd(float(row["jan_may_2026_max_dd_usd"])),
                mar_pnl=usd(float(row["march_2026_pnl_usd"])),
                mar_dd=usd(float(row["march_2026_max_dd_usd"])),
                d30_trades=int(row["last_30d_trades"]),
                d30_pnl=usd(float(row["last_30d_pnl_usd"])),
                d30_dd=usd(float(row["last_30d_max_dd_usd"])),
            )
        )
    return "\n".join(rows)


def executive_variant_rows(df: pd.DataFrame) -> str:
    decision_map = {
        "long_only_no_st": "Control baseline",
        "long_only_st5_50": "P0 candidate",
        "long_short_no_st": "Rejected as primary",
        "long_short_st5_50_aligned": "Exploratory only",
    }
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "| {label} | {trades:,} | {pnl} | {dd} | {retdd} | {mar_pnl} | {d30_pnl} | {decision} |".format(
                label=row["label"],
                trades=int(row["trades"]),
                pnl=usd(float(row["pnl_usd"])),
                dd=usd(float(row["max_dd_usd"])),
                retdd=maybe_num(row["return_dd"]),
                mar_pnl=usd(float(row["march_2026_pnl_usd"])),
                d30_pnl=usd(float(row["last_30d_pnl_usd"])),
                decision=decision_map.get(row["variant_id"], "Review"),
            )
        )
    return "\n".join(rows)


def build_executive_variant_snapshot() -> str:
    variant_df = safe_read_csv(ST_VARIANT_CSV)
    if variant_df is None:
        return """SuperTrend variant comparison belum tersedia saat report ini dibuat. Baseline
tetap menjadi satu-satunya measured strategy di executive summary ini.
"""

    return f"""Ringkasan varian utama:

| Variant | Trades | PnL | DD | Ret/DD | Mar 2026 PnL | 30D PnL | Current Use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{executive_variant_rows(variant_df)}

Keputusan sementara dari comparison ini:

- Baseline `Long only, no ST` tetap menjadi **control strategy** karena paling
  mudah diaudit dan 30D terakhir masih paling kuat.
- `Long only + ST5_50 bullish` menjadi **P0 candidate** untuk regime filter:
  drawdown full-history dan March 2026 membaik dengan hanya satu rule tambahan.
- `Long+Short, no ST` tidak dipromosikan karena short side mentah menambah
  frekuensi tetapi menurunkan kualitas risk-adjusted.
- `Long+Short + ST5_50 aligned` tetap ditrack sebagai exploratory variant:
  full-history dan March terlihat bagus, tetapi 30D terakhir negatif.
"""


def cross_variant_metric_rows() -> str:
    rows = []
    st_df = safe_read_csv(ST_VARIANT_CSV)
    if st_df is not None:
        wanted = [
            ("long_only_no_st", "Control"),
            ("long_only_st5_50", "P0 candidate"),
            ("long_short_no_st", "Rejected"),
            ("long_short_st5_50_aligned", "Exploratory"),
        ]
        for variant_id, role in wanted:
            matched = st_df[st_df["variant_id"] == variant_id]
            if matched.empty:
                continue
            row = matched.iloc[0]
            rows.append(
                "| {label} | {role} | {trades:,} | {pf:.2f} | {pnl} | {dd} | {mar_pnl} | {d30_pnl} |".format(
                    label=row["label"],
                    role=role,
                    trades=int(row["trades"]),
                    pf=float(row["profit_factor"]),
                    pnl=usd(float(row["pnl_usd"])),
                    dd=usd(float(row["max_dd_usd"])),
                    mar_pnl=usd(float(row["march_2026_pnl_usd"])),
                    d30_pnl=usd(float(row["last_30d_pnl_usd"])),
                )
            )

    switch_df = safe_read_csv(SHORT_SWITCH_CSV)
    if switch_df is not None:
        matched = switch_df[switch_df["variant_id"] == "short_switch_tp2r"]
        if not matched.empty:
            row = matched.iloc[0]
            rows.append(
                "| {label} | Watchlist | {trades:,} | {pf:.2f} | {pnl} | {dd} | {mar_pnl} | {d30_pnl} |".format(
                    label=row["label"],
                    trades=int(row["trades"]),
                    pf=float(row["profit_factor"]),
                    pnl=usd(float(row["pnl_usd"])),
                    dd=usd(float(row["max_dd_usd"])),
                    mar_pnl=usd(float(row["march_2026_pnl_usd"])),
                    d30_pnl=usd(float(row["last_30d_pnl_usd"])),
                )
            )

    if not rows:
        return "Belum ada cross-variant table. Jalankan audit SuperTrend dan short-switch terlebih dahulu."

    return "\n".join(rows)


def top_st_filter_rows(df: pd.DataFrame, limit: int = 8) -> str:
    liquid = df[
        (df["candidate"] != "BASELINE")
        & (df["full_trades"] >= 100)
        & (df["jan_may_2026_trades"] >= 30)
    ].copy()
    if liquid.empty:
        return ""
    liquid = liquid.sort_values(
        ["full_return_dd", "jan_may_2026_pnl"],
        ascending=[False, False],
    )
    rows = []
    for _, row in liquid.head(limit).iterrows():
        rows.append(
            "| {candidate} | {n} | {trades:,} | {pnl} | {dd} | {retdd} | {jm_trades:,} | {jm_pnl} | {mar_pnl} | {mar_dd} | {d30_pnl} |".format(
                candidate=row["candidate"],
                n=int(row["filter_count"]),
                trades=int(row["full_trades"]),
                pnl=usd(float(row["full_pnl"])),
                dd=usd(float(row["full_max_dd"])),
                retdd=maybe_num(row["full_return_dd"]),
                jm_trades=int(row["jan_may_2026_trades"]),
                jm_pnl=usd(float(row["jan_may_2026_pnl"])),
                mar_pnl=usd(float(row["march_2026_pnl"])),
                mar_dd=usd(float(row["march_2026_max_dd"])),
                d30_pnl=usd(float(row["last_30d_pnl"])),
            )
        )
    return "\n".join(rows)


def short_switch_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "| {label} | {trades:,} | {wr} | {pnl} | {dd} | {retdd} | {shorts:,} | {switches:,} | {short_pnl} | {jm_pnl} | {mar_pnl} | {d30_pnl} | {d30_dd} |".format(
                label=row["label"],
                trades=int(row["trades"]),
                wr=pct(float(row["win_rate"])),
                pnl=usd(float(row["pnl_usd"])),
                dd=usd(float(row["max_dd_usd"])),
                retdd=maybe_num(row["return_dd"]),
                shorts=int(row["short_first_trades"]),
                switches=int(row["switch_count"]),
                short_pnl=usd(float(row["short_pnl_usd"])),
                jm_pnl=usd(float(row["jan_may_2026_pnl_usd"])),
                mar_pnl=usd(float(row["march_2026_pnl_usd"])),
                d30_pnl=usd(float(row["last_30d_pnl_usd"])),
                d30_dd=usd(float(row["last_30d_max_dd_usd"])),
            )
        )
    return "\n".join(rows)


def p0_short_switch_row(row: pd.Series, label: str | None = None) -> str:
    return (
        "| {label} | {filter_id} | {short_risk} | {switch_risk} | {buffer} | {trades:,} | {pnl} | {dd} | {retdd} | {mar_pnl} | {d30_trades:,} | {d30_pnl} | {d30_dd} |".format(
            label=label or row["variant_id"],
            filter_id=row["short_filter"],
            short_risk=usd(float(row["short_risk_usd"])) if float(row["short_risk_usd"]) else "-",
            switch_risk=usd(float(row["switch_long_risk_usd"])) if float(row["switch_long_risk_usd"]) else "-",
            buffer=row["switch_buffer_mode"],
            trades=int(row["trades"]),
            pnl=usd(float(row["pnl_usd"])),
            dd=usd(float(row["max_dd_usd"])),
            retdd=maybe_num(row["return_dd"]),
            mar_pnl=usd(float(row["march_2026_pnl_usd"])),
            d30_trades=int(row["last_30d_trades"]),
            d30_pnl=usd(float(row["last_30d_pnl_usd"])),
            d30_dd=usd(float(row["last_30d_max_dd_usd"])),
        )
    )


def p0_short_switch_top_rows(df: pd.DataFrame, limit: int = 10) -> str:
    variants = df[~df["variant_id"].eq("long_only_no_st")].copy()
    if variants.empty:
        return ""
    variants = variants.sort_values(
        ["p0_recent_score", "last_30d_pnl_usd", "last_30d_max_dd_usd"],
        ascending=[False, False, False],
    )
    return "\n".join(p0_short_switch_row(row) for _, row in variants.head(limit).iterrows())


def build_p0_short_switch_section() -> str:
    p0_df = safe_read_csv(SHORT_SWITCH_P0_CSV)
    manifest = safe_read_json(SHORT_SWITCH_P0_MANIFEST)
    if p0_df is None:
        return """### 12.5 P0 Short-Switch TP2R Optimization Sweep

P0 short-switch sweep belum tersedia saat report ini dibuat. Jalankan:

```bash
python3 pipeline/mnq_ml/experiments/ORB/sweep_short_switch_tp2r_p0.py --force --short-entry-until none
```
"""

    baseline = p0_df[p0_df["variant_id"].eq("long_only_no_st")]
    existing = p0_df[p0_df["variant_id"].eq("p0_none_sr500_lr500_buf0_tgnone")]
    variants = p0_df[~p0_df["variant_id"].eq("long_only_no_st")].copy()
    if variants.empty or baseline.empty:
        return "### 12.5 P0 Short-Switch TP2R Optimization Sweep\n\nP0 summary kosong.\n"
    best = variants.sort_values(
        ["p0_recent_score", "last_30d_pnl_usd", "last_30d_max_dd_usd"],
        ascending=[False, False, False],
    ).iloc[0]
    raw_30d = variants.sort_values(
        ["last_30d_pnl_usd", "last_30d_max_dd_usd", "march_2026_pnl_usd"],
        ascending=[False, False, False],
    ).iloc[0]
    best_retdd = variants.sort_values(
        ["return_dd", "last_30d_pnl_usd"],
        ascending=[False, False],
    ).iloc[0]
    baseline_row = baseline.iloc[0]
    existing_row = existing.iloc[0] if not existing.empty else None

    variant_count = int((manifest or {}).get("rows", {}).get("summary", len(p0_df)) - 1)
    params = (manifest or {}).get("params", {})
    guard_values = ", ".join(params.get("short_entry_until", [])) or "n/a"
    lookahead_total = int(p0_df["short_filter_lookahead_violations"].fillna(0).sum())
    max_lag = p0_df["short_filter_max_lag_minutes"].max()

    comparison_rows = [p0_short_switch_row(baseline_row, "Long only baseline")]
    if existing_row is not None:
        comparison_rows.append(p0_short_switch_row(existing_row, "Existing short-switch TP2R equivalent"))
    comparison_rows.extend(
        [
            p0_short_switch_row(best, "Best P0 score"),
            p0_short_switch_row(raw_30d, "Best raw 30D PnL"),
            p0_short_switch_row(best_retdd, "Best full Ret/DD"),
        ]
    )

    return f"""### 12.5 P0 Short-Switch TP2R Optimization Sweep

P0 sweep ini memperbaiki branch `Short switch to long, short TP 2R` dengan
knob yang masih rule-based: short-side SuperTrend filter, asymmetric short risk,
switch-long risk, dan switch trigger buffer. Long-first breakout tetap memakai
baseline risk $500. Jika short breakout ditolak filter, hari tersebut masih
boleh mengambil later long breakout.

| Field | Value |
| --- | --- |
| Variants evaluated | {variant_count:,} plus baseline |
| Short filters | `none`, `st5_50_bearish`, `st5_20_bearish`, `st5_50_and_st15_20_bearish` |
| Short risk grid | $250, $350, $500 |
| Switch long risk grid | $500, $750 |
| Switch buffers | `0`, `2ticks`, `0.25r` |
| Short time guard grid | `{guard_values}` |
| Lookahead violations | {lookahead_total:,} |
| Max ST feature lag | {maybe_num(max_lag)} minutes |

| Candidate | Short Filter | Short Risk | Switch Long Risk | Buffer | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(comparison_rows)}

Top P0-score candidates:

| Candidate | Short Filter | Short Risk | Switch Long Risk | Buffer | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{p0_short_switch_top_rows(p0_df, limit=10)}

Current read:

- Best P0-score candidate: `{best["variant_id"]}`.
- 30D PnL membaik dari {usd(float(baseline_row["last_30d_pnl_usd"]))}
  menjadi {usd(float(best["last_30d_pnl_usd"]))}, dengan 30D DD tetap
  {usd(float(best["last_30d_max_dd_usd"]))}.
- March 2026 membaik dari {usd(float(baseline_row["march_2026_pnl_usd"]))}
  menjadi {usd(float(best["march_2026_pnl_usd"]))}.
- Full-history PnL naik, tetapi full-history DD memburuk dari
  {usd(float(baseline_row["max_dd_usd"]))} menjadi {usd(float(best["max_dd_usd"]))}.
  Jadi P0 result **membaik untuk Topstep-style recent window**, tetapi belum
  boleh dipromosikan sebelum P1 Topstep simulator dan time-guard sweep.

Artifact P0:

- `short_switch_tp2r_p0_sweep.md`
- `short_switch_tp2r_p0_full_report.md`
- `short_switch_tp2r_p0_sweep.csv`
- `short_switch_tp2r_p0_best_events.csv`
- `short_switch_tp2r_p0_best_legs.csv`
- `short_switch_tp2r_p0_best_yearly.csv`
- `short_switch_tp2r_p0_best_monthly.csv`
- `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_sweep_manifest.json`
"""


def build_short_switch_section() -> str:
    summary_df = safe_read_csv(SHORT_SWITCH_CSV)
    manifest = safe_read_json(SHORT_SWITCH_MANIFEST)
    if summary_df is None:
        return """## 12. Short Breakout Switch-To-Long Audit

Short switch audit belum tersedia saat report ini dibuat. Jalankan:

```bash
python3 pipeline/mnq_ml/experiments/ORB/build_short_reversal_switch_comparison.py --force
```

---
"""

    anchor = manifest["anchor_ts"] if manifest else "n/a"
    return f"""## 12. Short Breakout Switch-To-Long Audit

Section ini menguji definisi short yang asimetris terhadap long. Karena NASDAQ
secara natural lebih long-biased, short tidak diperlakukan sebagai mirror
strategy. Jika OR low break lebih dulu, strategy boleh masuk short; tetapi jika
harga close kembali di atas OR high, short ditutup dan posisi dibalik menjadi
long pada open M1 berikutnya.

### 12.1 Methodology

| Field | Value |
| --- | --- |
| Short entry | First M1 close below OR low |
| Short exit | TP 1R / 1.5R / 2R, OR switch to long, OR 15:00 NY EOD |
| Switch trigger | First M1 close above OR high while short is active |
| Switch execution | Close short and open long at next M1 open |
| Long after switch | Baseline long TP 2R or 15:00 NY EOD |
| Anchor | {anchor} |

### 12.2 Visual Audit

#### Equity Curve

![Short Switch Equity]({raw("charts/short_reversal_switch_equity_curve.png")})

#### Drawdown Curve

![Short Switch Drawdown]({raw("charts/short_reversal_switch_drawdown_curve.png")})

#### Last 30D Equity

![Short Switch Last 30D]({raw("charts/short_reversal_switch_last30_equity.png")})

### 12.3 Summary

| Variant | Trades | WR | PnL | DD | Ret/DD | Short-first | Switches | Short PnL | Jan-May PnL | Mar PnL | 30D PnL | 30D DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{short_switch_rows(summary_df)}

### 12.4 Current Read

Di antara varian short-switch, short TP 2R adalah yang paling kuat: total PnL
dan return/DD terbaik, serta short leg full-history positif. Namun ia masih
belum mengalahkan baseline pada window 30D terakhir dan max drawdown-nya masih
sedikit lebih berat dari baseline. Jadi short-switch TP 2R layak masuk watchlist
sebagai research branch, tetapi belum menggantikan long-only baseline.

{build_p0_short_switch_section()}

---
"""


def build_supertrend_section() -> str:
    variant_df = safe_read_csv(ST_VARIANT_CSV)
    filter_df = safe_read_csv(ST_FILTER_CSV)
    regime_manifest = safe_read_json(ST_REGIME_MANIFEST)
    variant_manifest = safe_read_json(ST_VARIANT_MANIFEST)

    if variant_df is None and filter_df is None:
        return """## 11. SuperTrend Regime Filter Audit

SuperTrend audit belum tersedia saat report ini dibuat. Jalankan:

```bash
python3 pipeline/mnq_ml/experiments/ORB/build_supertrend_regime_features.py --force
python3 pipeline/mnq_ml/experiments/ORB/build_supertrend_variant_comparison.py --force
```

---
"""

    lookahead_violations = 0
    max_lag = None
    feature_names = []
    if regime_manifest:
        lookahead_violations += int(regime_manifest["lookahead"]["total_violations"])
        max_lag = regime_manifest["lookahead"]["max_lag_minutes"]
        feature_names = regime_manifest["features"]["feature_names"]
    if variant_manifest:
        lookahead_violations += int(variant_manifest["lookahead_violations"])

    feature_text = ", ".join(f"`{name}`" for name in feature_names) if feature_names else "`ST5_50`"
    max_lag_text = f"{max_lag:.0f} menit" if max_lag is not None else "n/a"

    variant_table = ""
    if variant_df is not None:
        variant_table = f"""
### 11.2 Perbandingan Variant Utama

#### Equity Curve

![ST5_50 Variant Equity Curve]({raw("charts/supertrend_variant_equity_curve.png")})

#### Drawdown Curve

![ST5_50 Variant Drawdown Curve]({raw("charts/supertrend_variant_drawdown_curve.png")})

#### Monthly PnL 2026

![ST5_50 Variant Monthly PnL 2026]({raw("charts/supertrend_variant_monthly_pnl_2026.png")})

#### Rolling Window PnL/DD

![ST5_50 Variant Rolling Windows]({raw("charts/supertrend_variant_rolling_windows.png")})

#### Trade PnL Distribution

![ST5_50 Variant Trade PnL Distribution]({raw("charts/supertrend_variant_trade_pnl_distribution.png")})

#### March 2026 Equity

![ST5_50 Variant March 2026 Equity]({raw("charts/supertrend_variant_march_2026_equity.png")})

| Variant | Trades | Long | Short | WR | PnL | DD | Ret/DD | Jan-May Trades | Jan-May PnL | Jan-May DD | Mar PnL | Mar DD | 30D Trades | 30D PnL | 30D DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{variant_rows(variant_df)}

Interpretasi:

- `Long only, ST5_50 bullish` adalah kandidat P0 paling bersih: hanya menambah
  satu rule regime filter, March 2026 membaik, dan sample size masih besar.
- `Long+Short, no ST` menambah frekuensi, tetapi short leg mentahnya tidak
  cukup kuat karena PnL full-history turun dan DD membesar.
- `Long+Short, ST5_50 aligned` menarik secara full-history dan March, tetapi
  30D terakhir negatif. Ini belum layak jadi kandidat utama tanpa investigasi
  stabilitas recent window.
"""

    filter_table = ""
    if filter_df is not None:
        filter_table = f"""
### 11.3 Kandidat Kombinasi SuperTrend

Tabel ini menampilkan kandidat terbaik berdasarkan full-history return/DD,
dengan minimum `full_trades >= 100` dan `jan_may_2026_trades >= 30`.

| Candidate | N | Full Trades | Full PnL | Full DD | Ret/DD | Jan-May Trades | Jan-May PnL | Mar PnL | Mar DD | 30D PnL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{top_st_filter_rows(filter_df)}

Catatan: kombinasi multi-filter dapat memperbaiki March drawdown secara besar,
tetapi trade count turun drastis. Untuk menghindari curve fitting, kandidat
yang lebih sederhana tetap diprioritaskan sebelum kombinasi kompleks.
"""

    return f"""## 11. SuperTrend Regime Filter Audit

SuperTrend audit ditambahkan untuk menjawab apakah drawdown March 2026 bisa
dikurangi dengan regime filter sederhana, tanpa langsung mengganti baseline.
Semua fitur dihitung dari bar yang sudah close dan di-join ke trade event
dengan rule `feature_ts <= signal_ts`.

### 11.1 Data Integrity

| Check | Value |
| --- | ---: |
| Feature family | {feature_text} |
| SuperTrend factor | 4.00 |
| Direction convention | `-1 = bullish/up`, `+1 = bearish/down` |
| Join rule | Latest completed feature timestamp `<= signal_ts` |
| Lookahead violations | {lookahead_violations:,} |
| Max feature lag | {max_lag_text} |

{variant_table}
{filter_table}
### 11.4 Keputusan Sementara SuperTrend

Untuk saat ini baseline **tidak diganti**. Baseline tetap `Long only, no ST`
sebagai control. Kandidat yang dibawa ke iterasi berikutnya:

1. `Long only + ST5_50 bullish` sebagai P0 regime-filter candidate.
2. `Long+Short + ST5_50 aligned` sebagai exploratory candidate, bukan prioritas
   utama, karena 30D terakhir masih negatif.

---
"""


def build_report(events: pd.DataFrame, summary: dict, mc: dict) -> str:
    perf = summary["performance"]
    quality = summary["daily_quality"]
    costs = summary["costs"]
    signal = summary["signal_range"]
    windows = summary["windows"]

    window_order = ["5D", "10D", "20D", "30D", "50D", "100D", "200D"]
    window_rows = "\n".join(
        "| {window} | {trades:,} | {wr} | {pnl} | {dd} |".format(
            window=window,
            trades=int(windows[window]["trades"]),
            wr=pct(windows[window]["win_rate"]),
            pnl=usd(windows[window]["pnl_usd"]),
            dd=usd(windows[window]["max_dd_usd"]),
        )
        for window in window_order
    )
    supertrend_section = build_supertrend_section()
    short_switch_section = build_short_switch_section()
    executive_variant_snapshot = build_executive_variant_snapshot()

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return f"""# Strategi NASDAQ Micro Futures Opening Range Breakout Rule-Based Iterasi v1
**Evaluasi Baseline 15m Long TP2R/EOD pada kontrak MNQ**

Tanggal laporan: **{created_at[:10]}**

Model / strategy ID: `rule_based_15m_long_tp2r_eod`

Objective: **Topstep 50K research baseline and regime-filter comparison** -
mencari apakah breakout NASDAQ Micro Futures setelah 15 menit pertama New York
open punya positive expectancy yang cukup untuk menjadi kandidat forward test,
lalu menilai apakah SuperTrend sederhana dapat memperbaiki drawdown tanpa
merusak recent performance.

Audience: trader futures, evaluator internal strategi NASDAQ futures, dan
pembanding untuk overlay machine learning.

---

## 1. Ringkasan Eksekutif

Laporan ini mengevaluasi strategi NASDAQ Micro Futures ORB v1 sebagai
**rule-based research package**. Baseline long-only tetap menjadi control,
tetapi report ini juga memuat comparison terhadap regime filter SuperTrend dan
eksplorasi long+short. Ticker teknis yang digunakan di data dan backtest adalah
`MNQ`, yaitu Micro E-mini Nasdaq-100 futures.

Aturan yang diuji sederhana: ambil posisi long setelah candle M1 pertama close
di atas high opening range 15 menit, entry pada open M1 berikutnya, lalu exit
di TP 2R atau time exit 15:00 New York. Strategi ini tidak memakai normal stop
loss; OR low hanya menjadi referensi sizing.

| Area | Hasil |
| --- | ---: |
| Periode sinyal | {signal["min_signal_ts"][:10]} - {signal["max_signal_ts"][:10]} |
| Baseline total trade | {int(perf["trades"]):,} |
| Baseline win rate | {pct(perf["win_rate"])} |
| Baseline net PnL | {usd(perf["total_pnl_usd"])} |
| Baseline max drawdown | {usd(perf["max_dd_usd"])} |
| Baseline profit factor | {perf["profit_factor"]:.2f} |
| Baseline daily Sharpe / Sortino | {quality["daily_sharpe_annualized"]:.2f} / {quality["daily_sortino_annualized"]:.2f} |
| Baseline 30D terakhir | {int(windows["30D"]["trades"])} trade, {usd(windows["30D"]["pnl_usd"])} PnL, {usd(windows["30D"]["max_dd_usd"])} max DD |

{executive_variant_snapshot}

**Kesimpulan utama:** strategi ini belum boleh dibaca sebagai satu final live
strategy. Baseline membuktikan ada continuation edge, terutama pada 30D
terakhir, tetapi long-run PF masih tipis dan max drawdown historis terlalu
besar untuk langsung masuk Topstep live. SuperTrend `ST5_50` memberi perbaikan
drawdown yang jelas, khususnya pada March 2026, namun menurunkan 30D PnL. Maka
keputusan institusional saat ini adalah: baseline tetap control, `Long only +
ST5_50` masuk P0 candidate, long+short ST aligned tetap exploratory, dan semua
variant perlu Topstep MLL/consistency simulator sebelum forward execution.

---

## 2. Latar Belakang Strategi

Opening Range Breakout berangkat dari hipotesis bahwa rentang harga pada awal
sesi New York menyimpan informasi tentang imbalance intraday. Untuk Nasdaq
futures, tekanan order setelah cash open sering menjadi penentu arah sesi.
Strategi ini mencari continuation setelah harga keluar dari opening range,
bukan mean reversion intraday.

### 2.1 Research Problem

Target riset bukan hanya mencari total PnL tertinggi. Untuk konteks Topstep 50K,
strategi harus menjawab beberapa pertanyaan praktis:

1. Apakah ORB NASDAQ Micro Futures 15m punya positive expectancy setelah biaya dan slippage?
2. Apakah edge cukup aktif untuk window evaluasi sekitar 30 hari?
3. Apakah drawdown masih masuk akal terhadap MLL dan consistency rule?
4. Apakah filter sederhana dapat mengurangi bulan buruk seperti March 2026
   tanpa menghapus trade terbaik pada April-May 2026?
5. Apakah sisi short menambah edge atau hanya menambah noise/frequency?

### 2.2 Why Baseline First

Versi ini sengaja dibuat sederhana:

1. Tidak memakai indikator tambahan.
2. Tidak memakai filter ML.
3. Tidak melakukan short continuation.
4. Tidak melakukan reversal.
5. Tidak memakai normal SL sebagai exit strategi.

Tujuannya adalah mendapatkan **baseline bersih**. Jika baseline saja tidak
punya edge, ML overlay akan mudah menjadi curve fitting. Jika baseline punya
edge, ML dapat diuji sebagai risk adjuster, bukan sebagai alasan untuk memaksa
trade.

Baseline long-only juga berfungsi sebagai control: setiap filter, ML model,
atau long+short extension harus mengalahkan baseline pada risk-adjusted metrics,
bukan hanya menaikkan satu angka PnL.

### 2.3 Why SuperTrend Was Added To The Audit

March 2026 menunjukkan kelemahan utama baseline: continuation long-only bisa
terjebak pada regime yang tidak mendukung breakout. SuperTrend diuji sebagai
regime filter karena:

- Rule-nya eksplisit dan mudah diaudit.
- Bisa dihitung dari bar yang sudah close, sehingga no-lookahead bisa digate.
- Mewakili trend state tanpa langsung menjadi model ML.
- Cocok sebagai risk filter sebelum masuk ke probability sizing.

Audit menghitung ST 5m/15m dengan ATR 5/10/20/50. Kandidat paling sederhana
yang muncul adalah `ST5_50`: long breakout hanya diambil saat ST5_50 bullish.

### 2.4 Why Long+Short Was Tested

Long+short diuji karena breakout bawah secara teori bisa memberi tambahan
frequency. Namun hasil awal menunjukkan short mentah tidak otomatis punya edge.
Ketika short disejajarkan dengan ST5_50 bearish, full-history membaik, tetapi
recent 30D memburuk. Karena itu long+short belum dipromosikan; ia tetap menjadi
exploratory branch yang perlu investigasi lanjutan.

### 2.5 Methodology Guardrails

Semua angka dalam report ini harus dibaca dengan guardrail berikut:

- Entry memakai signal close M1, lalu entry di open M1 berikutnya.
- SuperTrend feature hanya boleh memakai timestamp fitur `<= signal_ts`.
- Biaya TopstepX MNQ dan slippage sudah masuk.
- Baseline dan varian ST adalah rule-based, bukan ML.
- Laporan ini research-only; belum live-ready.

---

## 3. Data Lineage & Control Gates

Ini bagian paling penting secara governance: **bronze/L0 data salah berarti
semua angka PnL, drawdown, chart, dan kesimpulan report ini ikut salah**.
Karena itu report ini harus dibaca dari bawah ke atas: L0 raw integrity,
L1 feature/context integrity, L2 event integrity, lalu baru performance.

### 3.1 Lineage

```text
Databento MNQ M1 + yfinance recent append
  -> data/Level_0_Raw/MNQ_1m.duckdb                 # bronze/raw canonical M1
  -> data/Level_1_Features/mnq/ORB/context.parquet  # M1 NY-session context + quality flags
  -> data/Level_2_Datamart/mnq/ORB/sweeps           # ORB grid events
  -> data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod
  -> model/MNQ/ORB/rule_based_15m_long_tp2r_eod
```

Baseline control report ini memakai frozen L2 event package
`rule_based_15m_long_tp2r_eod/events.parquet`. SuperTrend dan short-switch
adalah audit/variant layer yang menempel ke baseline/sweep events, bukan data
source baru.

### 3.2 Gate Summary

| Layer | Artifact | Control Result |
| --- | --- | --- |
{data_lineage_control_rows(events)}

### 3.3 Feature Dataset Inventory

| Layer | Dataset | Content | Shape | Role |
| --- | --- | --- | --- | --- |
{data_feature_dataset_rows(events)}

### 3.4 Lookahead Contract

| Component | Contract |
| --- | --- |
| ORB baseline | Signal is M1 candle close; execution is next M1 open, so `entry_ts > signal_ts` is mandatory. |
| Opening range | OR high/low use only completed bars inside the first N minutes after 09:30 NY. |
| Intraday quality | Trade entry/exit scans use `bar_data_quality_ok`; bars containing source gaps are excluded from decision scanning. |
| SuperTrend attach | Feature timestamp must be `<= signal_ts`; current attached lookahead violations = 0. |
| Daily confluence | External daily features use date `< ny_date`; current daily confluence lookahead violations = 0. |
| Labels/PnL | Exit, PnL, R multiple, and realized outcome are labels/evaluation fields, not allowed as pre-trade features. |

### 3.5 Current Data Risk Read

- L0 hard integrity is clean: no duplicate timestamps, null OHLCV, bad OHLC, or
  negative volume rows in the latest audit.
- L0 continuity is **not perfectly gap-free**. The current status is
  `PASS_WITH_GAPS_REQUIRING_L1_QUARANTINE`, so the correct interpretation is
  not "data has no gaps", but "known gaps are flagged and downstream builders
  must not train or trade through bad bars".
- L1 context audit passes and explicitly checks that OR values do not appear on
  pre-OR rows, no eligible rows exist before OR end, and eligible rows are not
  bad-quality bars.
- L2 baseline events pass event-level controls used by this report: no required
  nulls, no duplicate NY dates, entry after signal, and exit after entry.
- Any future ML model must promote these gates to hard blockers. A model trained
  on failed L0/L1/L2 gates is invalid even if its backtest looks profitable.

---

## 4. Konteks Strategy Family

Report ini tidak lagi hanya berisi satu baseline long-only. Scope saat ini
adalah **family of rule-based ORB variants** untuk NASDAQ Micro Futures, dengan
baseline sebagai control dan beberapa branch sebagai kandidat riset.

### 4.1 Common Contract

| Field | Value |
| --- | --- |
| Instrument | NASDAQ Micro Futures (`MNQ`) |
| Session | New York regular session |
| Source grain | Right-labeled M1 bars |
| Opening range | First 15 minutes after 09:30 NY |
| Entry timing | Signal after M1 close, execution on next M1 open |
| Primary exit clock | 15:00 NY EOD/time exit |
| Cost model | TopstepX MNQ commission + 1 tick slippage per side |
| Target risk | $500 |
| Baseline max trades | 1 sequence per NY session |
| Research status | Not live-ready |

### 4.2 Variant Map

| Variant | Role | Direction Logic | Regime Filter | Exit Logic | Current Status |
| --- | --- | --- | --- | --- | --- |
| Long only, no ST | Control baseline | First M1 close above OR high | None | TP 2R or 15:00 NY | Keep as benchmark |
| Long only + ST5_50 | P0 candidate | Same as baseline | Require ST5_50 bullish at signal close | TP 2R or 15:00 NY | Best simple regime filter |
| Long+Short, no ST | Rejected as primary | First breakout either OR high or OR low | None | TP 2R or 15:00 NY | Adds frequency but weak risk-adjusted quality |
| Long+Short + ST5_50 aligned | Exploratory | Long with bullish ST, short with bearish ST | ST5_50 aligned by side | TP 2R or 15:00 NY | Good March, weak recent 30D |
| Short switch to long | Research branch | Short if OR low breaks first; switch to long if OR high reclaimed | None in current test | Short TP 1R/1.5R/2R, switch, or EOD | TP 2R best, not promoted |

### 4.3 Rule-Based Definition

Semua variant di report ini masih **rule-based**, bukan ML. Keputusan entry,
exit, switch, dan filter ditentukan oleh aturan eksplisit. Belum ada model
probabilitas yang menentukan trade size, trade/no-trade, atau direction.

### 4.4 Current Promotion Hierarchy

1. **Benchmark:** `Long only, no ST`.
2. **P0 candidate:** `Long only + ST5_50 bullish`.
3. **Watchlist:** `Short switch to long, short TP 2R`.
4. **Exploratory only:** `Long+Short + ST5_50 aligned`.
5. **Not promoted:** `Long+Short, no ST`.

---

## 5. Sizing dan Risk Model

Semua varian di report ini memakai target risk dollar tetap sebagai sizing
anchor. Target risk bukan normal stop-loss order; ia hanya menentukan jumlah
kontrak berdasarkan jarak entry ke referensi opening range.

```text
contracts_float = target_risk_usd / risk_per_contract_usd
contracts_used = floor(contracts_float), minimum 1 contract
```

Dengan `target_risk_usd = $500`, jumlah kontrak otomatis turun saat OR/risk
melebar dan naik saat risk menyempit. Karena kontrak harus integer, actual risk
tidak selalu tepat $500.

Referensi risk per family:

| Family | Risk Reference | Exit |
| --- | --- | --- |
| Long only baseline | Entry minus OR low | TP 2R atau 15:00 NY |
| Long only + ST5_50 | Entry minus OR low | TP 2R atau 15:00 NY |
| Short continuation | OR high minus entry | TP 2R atau 15:00 NY |
| Short switch to long | Short leg memakai OR high; switched long memakai OR low | Short TP/switch/EOD, lalu long TP 2R/EOD |

Catatan penting: OR high/low **bukan** normal stop loss strategi. Exit loss
utama tetap time exit, sehingga live version tetap membutuhkan catastrophic
guard terpisah.

---

## 6. Baseline Control - Equity, Drawdown, Monthly PnL

Section ini hanya untuk baseline control `Long only, no ST`. Tujuannya adalah
menyediakan benchmark bersih sebelum membaca audit SuperTrend dan short-switch
di section 11-12.

### 6.1 Equity Curve

![Equity Curve]({raw("charts/equity_curve.png")})

Equity curve baseline menunjukkan PnL positif secara historis, tetapi jalurnya
tidak linear. Ada fase panjang yang relatif datar dan beberapa periode drawdown
besar.

### 6.2 Drawdown

![Drawdown Curve]({raw("charts/drawdown_curve.png")})

Drawdown maksimum historis sebesar {usd(perf["max_dd_usd"])}. Ini jauh lebih
besar daripada batas MLL Topstep 50K, sehingga evaluasi live tidak boleh hanya
mengandalkan total PnL historis.

### 6.3 Monthly PnL

![Monthly PnL]({raw("charts/monthly_pnl.png")})

Grafik bulanan baseline membantu melihat bahwa strategi tidak menghasilkan
distribusi profit yang stabil setiap bulan. Ada bulan kuat, bulan kosong, dan
bulan rugi.

### 6.4 Distribusi PnL Per Trade

![Trade PnL Distribution]({raw("charts/trade_pnl_distribution.png")})

Pada baseline, rata-rata loss per trade masih lebih besar daripada rata-rata
win. Edge muncul dari kombinasi win rate 56.48%, sizing, dan beberapa periode
momentum yang produktif.

---

## 7. Baseline Control - Performance Card dan Variant Snapshot

Section 7.1 adalah metric card baseline `Long only, no ST`. Ini bukan metric
untuk seluruh strategy family. Cross-variant context langsung ditaruh di
section 7.2 agar baseline, ST filter, dan short-switch tidak tercampur.

### 7.1 Baseline Control Metrics

| Metric | Value |
| --- | ---: |
| Signal range | {signal["min_signal_ts"][:10]} to {signal["max_signal_ts"][:10]} |
| Trades | {int(perf["trades"]):,} |
| Win rate | {pct(perf["win_rate"])} |
| Net PnL | {usd(perf["total_pnl_usd"])} |
| Gross profit | {usd(perf["gross_profit_usd"])} |
| Gross loss | {usd(perf["gross_loss_usd"])} |
| Profit factor | {perf["profit_factor"]:.2f} |
| Max drawdown | {usd(perf["max_dd_usd"])} |
| Return / DD | {perf["return_dd"]:.2f} |
| Expectancy / trade | {usd(perf["expectancy_per_trade_usd"])} |
| Median trade | {usd(perf["median_pnl_usd"])} |
| Average win | {usd(perf["avg_win_usd"])} |
| Average loss | {usd(perf["avg_loss_usd"])} |
| Payoff ratio | {perf["payoff_ratio"]:.2f} |
| Average contracts | {perf["avg_contracts"]:.2f} |
| Max consecutive wins | {int(perf["max_consecutive_wins"])} |
| Max consecutive losses | {int(perf["max_consecutive_losses"])} |

### 7.2 Cross-Variant Metric Snapshot

| Variant | Role | Trades | PF | Full PnL | Max DD | Mar 2026 PnL | 30D PnL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{cross_variant_metric_rows()}

Reading note:

- Baseline still has the strongest recent 30D PnL.
- `Long only + ST5_50` improves full-history drawdown and March 2026, but gives
  up some recent upside.
- Short-switch TP2R improves full-history PnL, but not enough on March/30D to
  replace the long-only control.

---

## 8. Cost Model

| Cost | Value |
| --- | ---: |
| Commission + fees | ${costs["commission_round_turn_usd_per_contract"]:.2f} RT / contract |
| Slippage | {costs["slippage_ticks_per_side"]} tick per side |
| Modeled slippage | ${costs["slippage_round_turn_usd_per_contract"]:.2f} RT / contract |
| Total commission paid | {usd(costs["total_commission_paid_usd"])} |
| Total modeled slippage | {usd(costs["total_modeled_slippage_usd"])} |

Biaya ini dipakai konsisten untuk baseline-control dan audit varian yang dibuat
dalam package ini. `pnl_net_usd` sudah memasukkan TopstepX MNQ, yaitu kontrak
Micro E-mini Nasdaq-100 futures, $1.24 round-turn per contract dan modeled
slippage 1 tick per side.

---

## 9. Baseline Control - Daily Quality

Daily quality di section ini hanya untuk baseline-control. Sharpe and Sortino
are computed from daily dollar PnL over NASDAQ Micro Futures NY session days,
with zero PnL on no-trade days, annualized by `sqrt(252)`.

| Metric | Value |
| --- | ---: |
| Trading days measured | {int(quality["trading_days"]):,} |
| Active days | {int(quality["active_days"]):,} |
| Active-day rate | {pct(quality["active_day_rate"])} |
| Active-day win rate | {pct(quality["active_day_win_rate"])} |
| Daily average PnL | {usd(quality["daily_avg_pnl_usd"])} |
| Daily PnL std dev | {usd(quality["daily_std_pnl_usd"])} |
| Daily Sharpe | {quality["daily_sharpe_annualized"]:.2f} |
| Daily Sortino | {quality["daily_sortino_annualized"]:.2f} |
| Best day | {usd(quality["best_day_pnl_usd"])} |
| Worst day | {usd(quality["worst_day_pnl_usd"])} |
| Best-day profit share | {pct(quality["best_day_profit_share"])} |
| 50% consistency flag | {"Pass" if quality["topstep_50pct_consistency_ok"] else "Fail"} |

Variant daily quality belum dijadikan decision metric utama karena varian ST dan
short-switch masih perlu Topstep-specific simulator yang sama sebelum promosi.

---

## 10. Baseline Control - Rolling Window Terakhir

![Rolling Windows]({raw("charts/rolling_windows.png")})

| Window | Trades | Win Rate | PnL | Max DD |
| ---: | ---: | ---: | ---: | ---: |
{window_rows}

Interpretasi baseline:

- 30D terakhir adalah bagian paling menarik: 18 trade dan {usd(windows["30D"]["pnl_usd"])} PnL.
- 5D dan 10D masih terlalu pendek untuk menjadi bukti edge.
- 100D dan 200D tetap positif, tetapi DD historisnya mulai berat untuk Topstep.
- Rolling comparison untuk varian ST dan short-switch ada di section 11-12,
  bukan di chart baseline ini.

---

{supertrend_section}

{short_switch_section}

## 13. Baseline Control - Monte Carlo dan Stress Test

Monte Carlo di section ini hanya memakai daily PnL baseline-control. Ini bukan
prediksi masa depan, dan belum boleh dibaca sebagai Monte Carlo untuk ST5_50
atau short-switch. Fungsinya adalah stress test distribusi baseline jika pola
daily PnL historis muncul dalam urutan yang berbeda.

| Horizon | Median PnL | P5 PnL | Prob. Akhir Rugi | Median MaxDD | Prob. DD <= -$2k | Prob. Hit +$3k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{monte_rows(mc)}

### 13.1 Fan Chart 30D

![Monte Carlo PnL Fan 30D]({raw("monte_carlo/monte_pnl_fan_30d.png")})

### 13.2 Distribusi Final PnL 30D

![Monte Carlo Final PnL CDF 30D]({raw("monte_carlo/monte_final_pnl_cdf_30d.png")})

### 13.3 Max Drawdown 30D

![Monte Carlo MaxDD 30D]({raw("monte_carlo/monte_maxdd_hist_30d.png")})

### 13.4 Fan Chart 100D

![Monte Carlo PnL Fan 100D]({raw("monte_carlo/monte_pnl_fan_100d.png")})

Kesimpulan Monte Carlo baseline: strategi punya upside untuk mencapai +$3,000
dalam sebagian path 30D, tetapi risiko drawdown terhadap batas -$2,000 tetap
perlu diuji lebih ketat dengan simulator Topstep yang memperhitungkan aturan
akun. Setelah simulator itu ada, baseline, ST5_50, dan short-switch TP2R harus
dibandingkan ulang dengan metodologi yang sama.

---

## 14. Penilaian Risiko

### 14.1 Risiko Drawdown

Baseline max drawdown historis {usd(perf["max_dd_usd"])} jauh lebih besar
daripada MLL Topstep 50K. ST5_50 menurunkan drawdown full-history, tetapi belum
menghapus risiko MLL karena window 30D dan intraday path tetap harus
disimulasikan. Ini tidak otomatis membatalkan strategi, tetapi semua varian
membutuhkan guard dan monitoring harian.

### 14.2 Risiko No Normal SL

Strategi ini tidak memakai SL normal. Exit loss terjadi lewat time exit.
Konsekuensinya, flash drop atau trend day yang berlawanan bisa menghasilkan
kerugian lebih besar dari target risk teoritis. Catastrophic guard harus
dipilih sebagai layer operasional terpisah.

### 14.3 Risiko Curve Fit

Baseline cukup bersih karena hanya memakai OR 15m, long only, TP 2R/time exit,
dan risk $500. Risiko curve fit naik pada kombinasi multi-SuperTrend dan
short-switch karena jumlah pilihan bertambah. Karena itu kandidat sederhana
`ST5_50` diprioritaskan atas kombinasi multi-filter walaupun beberapa kombinasi
punya return/DD historis lebih tinggi.

### 14.4 Risiko Eksekusi Live

Live version harus memastikan:

- M1 candle close sudah final sebelum entry.
- Entry dilakukan pada open M1 berikutnya.
- Jam New York dan daylight saving benar.
- Tidak ada duplicate trade per hari.
- Tidak ada posisi tanpa catastrophic guard.
- Data feed dan broker connection punya heartbeat.

---

## 15. Rekomendasi Sementara

| Area | Rekomendasi |
| --- | --- |
| Baseline control | Pertahankan `Long only, no ST` sebagai benchmark wajib |
| P0 regime filter | Bawa `Long only + ST5_50 bullish` ke Topstep simulator |
| Short branch | Track `Short switch to long, short TP 2R`, tetapi jangan promosi dulu |
| Live trading | Belum live-ready |
| Forward test | Baru layak paper/forward-test setelah simulator MLL/consistency selesai |
| ML overlay | Hanya boleh menjadi risk adjuster, bukan filter trade utama dulu |
| Sizing default | Tetap $500 sampai MLL/consistency simulator selesai |
| Guard | Wajib desain catastrophic guard sebelum live |

Rekomendasi utama:

1. Kunci baseline sebagai control, bukan final live strategy.
2. Bandingkan baseline vs `Long only + ST5_50` vs `Short switch TP2R` memakai
   Topstep-specific simulator yang sama.
3. Jangan mengganti baseline dengan ML sebelum ML terbukti memperbaiki
   risk-adjusted return dan sizing decision terhadap control ini.
4. Prioritas berikutnya adalah simulator: MLL, consistency, first +$3,000 path,
   daily loss guard, dan catastrophic guard.

---

## 16. Keputusan Sementara

| Area | Status |
| --- | --- |
| Strategy family | Rule-based ORB research package |
| Baseline edge | Ada, tetapi PF masih tipis |
| P0 candidate | `Long only + ST5_50 bullish` |
| Watchlist | `Short switch to long, short TP 2R` |
| 30D Topstep-style potential | Menarik pada baseline, belum cukup tanpa simulator |
| Long-run robustness | Perlu guard, regime review, dan Topstep path sim |
| Live readiness | Belum |
| Model package | Siap sebagai institutional-style research report |

Keputusan sementara: **package ini dipertahankan sebagai NASDAQ Micro Futures
ORB rule-based strategy family v1**. Baseline tetap control, ST5_50 menjadi P0
candidate, short-switch TP2R menjadi watchlist. Belum ada approval untuk live
execution.

---

## 17. Artifact Register

### 17.1 Model Package

| File | Keterangan |
| --- | --- |
| `README.md` | Model card singkat |
| `REPORT.md` | Laporan utama ini |
| `metrics.json` | Ringkasan metrik machine-readable |
| `manifest.json` | Lineage source/output |
| `charts/equity_curve.png` | Baseline-control equity curve |
| `charts/drawdown_curve.png` | Baseline-control drawdown curve |
| `charts/monthly_pnl.png` | Baseline-control monthly PnL |
| `charts/rolling_windows.png` | Baseline-control rolling window PnL/DD |
| `charts/trade_pnl_distribution.png` | Baseline-control distribusi PnL trade |
| `charts/supertrend_variant_equity_curve.png` | Equity curve perbandingan varian ST5_50 |
| `charts/supertrend_variant_drawdown_curve.png` | Drawdown curve perbandingan varian ST5_50 |
| `charts/supertrend_variant_monthly_pnl_2026.png` | Monthly PnL 2026 perbandingan varian ST5_50 |
| `charts/supertrend_variant_rolling_windows.png` | Rolling PnL/DD perbandingan varian ST5_50 |
| `charts/supertrend_variant_trade_pnl_distribution.png` | Distribusi trade PnL perbandingan varian ST5_50 |
| `charts/supertrend_variant_march_2026_equity.png` | Equity khusus March 2026 perbandingan varian ST5_50 |
| `charts/short_reversal_switch_equity_curve.png` | Equity curve varian short-switch-to-long |
| `charts/short_reversal_switch_drawdown_curve.png` | Drawdown curve varian short-switch-to-long |
| `charts/short_reversal_switch_last30_equity.png` | Last 30D equity varian short-switch-to-long |
| `monte_carlo/monte_pnl_fan_30d.png` | Baseline-control Monte Carlo fan chart 30D |
| `monte_carlo/monte_final_pnl_cdf_30d.png` | Baseline-control Monte Carlo final PnL CDF 30D |
| `monte_carlo/monte_maxdd_hist_30d.png` | Baseline-control Monte Carlo MaxDD histogram 30D |
| `monte_carlo/monte_pnl_fan_100d.png` | Baseline-control Monte Carlo fan chart 100D |
| `supertrend_regime_audit.md` | Audit grid SuperTrend 5m/15m ATR 5/10/20/50 |
| `supertrend_filter_candidates.csv` | Semua kandidat kombinasi bullish SuperTrend |
| `supertrend_variant_comparison.md` | Perbandingan baseline, ST5_50, long+short, dan long+short ST aligned |
| `supertrend_variant_comparison.csv` | Tabel machine-readable untuk perbandingan variant ST5_50 |
| `short_reversal_switch_comparison.md` | Audit short breakout yang switch ke long saat OR high reclaim |
| `short_reversal_switch_comparison.csv` | Summary varian short TP 1R/1.5R/2R |
| `short_reversal_switch_events.csv` | Sequence-level event varian short-switch |
| `short_reversal_switch_legs.csv` | Leg-level attribution varian short-switch |
| `short_switch_tp2r_p0_sweep.md` | P0 sweep short-switch TP2R dengan ST filter, asymmetric risk, dan switch buffer |
| `short_switch_tp2r_p0_full_report.md` | Full report kandidat P0 terbaik: yearly, YTD, month-to-month |
| `short_switch_tp2r_p0_sweep.csv` | Summary machine-readable P0 short-switch TP2R |
| `short_switch_tp2r_p0_best_events.csv` | Sequence-level event kandidat P0 terbaik |
| `short_switch_tp2r_p0_best_legs.csv` | Leg-level attribution kandidat P0 terbaik |
| `short_switch_tp2r_p0_best_yearly.csv` | Yearly metrics kandidat P0 terbaik |
| `short_switch_tp2r_p0_best_monthly.csv` | Monthly metrics kandidat P0 terbaik |

### 17.2 Canonical Data

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/summary.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/package_gate.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_regime_features.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_regime_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_variant_comparison_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_reversal_switch_comparison_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_sweep_manifest.json
data/Level_0_Raw/MNQ_1m.duckdb
data/Level_0_Raw/MNQ_1m_duckdb_manifest.json
data/Level_0_Raw/MNQ_1m_yfinance_append_manifest.json
data/Level_0_Raw/MNQ_1m_continuity_report.json
data/Level_0_Raw/MNQ_yfinance_timeframe_parity_report.json
data/Level_1_Features/mnq/ORB/context.parquet
data/Level_1_Features/mnq/ORB/context_manifest.json
data/Level_1_Features/mnq/ORB/l1_audit.json
data/Level_1_Features/mnq/ORB/daily_confluence.parquet
data/Level_1_Features/mnq/ORB/daily_confluence_manifest.json
data/Level_1_Features/mnq/ORB/daily_confluence_audit.json
data/Level_2_Datamart/mnq/ORB/sweeps/sweep_events.parquet
data/Level_2_Datamart/mnq/ORB/sweeps/sweep_manifest.json
```

---

## 18. Lampiran A - 10 Trade Terakhir

| NY Date | Signal UTC | Exit | Contracts | Net PnL |
| --- | --- | --- | ---: | ---: |
{last_trades_rows(events)}
"""


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    MONTE_DIR.mkdir(parents=True, exist_ok=True)

    events, summary = load_inputs()
    daily_pnl = build_daily_pnl(events)
    mc = monte_carlo(daily_pnl)

    save_equity_curve(events)
    save_drawdown_curve(events)
    save_monthly_pnl(events)
    save_rolling_windows(summary)
    save_trade_pnl_distribution(events)
    save_monte_carlo_charts(mc)

    (MODEL_DIR / "REPORT.md").write_text(build_report(events, summary, mc))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    serializable_mc = {
        key: {k: v for k, v in value.items() if k not in {"final_pnl", "max_dd", "sample_paths"}}
        for key, value in mc.items()
    }
    (MODEL_DIR / "monte_carlo_metrics.json").write_text(
        json.dumps(serializable_mc, indent=2, sort_keys=True) + "\n"
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy_id": summary["strategy_id"],
        "source_events": rel(DATA_DIR / "events.parquet"),
        "source_summary": rel(DATA_DIR / "summary.json"),
        "outputs": {
            "report": rel(MODEL_DIR / "REPORT.md"),
            "metrics": rel(MODEL_DIR / "metrics.json"),
            "charts": [
                rel(CHART_DIR / "equity_curve.svg"),
                rel(CHART_DIR / "equity_curve.png"),
                rel(CHART_DIR / "drawdown_curve.svg"),
                rel(CHART_DIR / "drawdown_curve.png"),
                rel(CHART_DIR / "monthly_pnl.svg"),
                rel(CHART_DIR / "monthly_pnl.png"),
                rel(CHART_DIR / "rolling_windows.svg"),
                rel(CHART_DIR / "rolling_windows.png"),
                rel(CHART_DIR / "trade_pnl_distribution.svg"),
                rel(CHART_DIR / "trade_pnl_distribution.png"),
                rel(CHART_DIR / "supertrend_variant_equity_curve.svg"),
                rel(CHART_DIR / "supertrend_variant_equity_curve.png"),
                rel(CHART_DIR / "supertrend_variant_drawdown_curve.svg"),
                rel(CHART_DIR / "supertrend_variant_drawdown_curve.png"),
                rel(CHART_DIR / "supertrend_variant_monthly_pnl_2026.svg"),
                rel(CHART_DIR / "supertrend_variant_monthly_pnl_2026.png"),
                rel(CHART_DIR / "supertrend_variant_rolling_windows.svg"),
                rel(CHART_DIR / "supertrend_variant_rolling_windows.png"),
                rel(CHART_DIR / "supertrend_variant_trade_pnl_distribution.svg"),
                rel(CHART_DIR / "supertrend_variant_trade_pnl_distribution.png"),
                rel(CHART_DIR / "supertrend_variant_march_2026_equity.svg"),
                rel(CHART_DIR / "supertrend_variant_march_2026_equity.png"),
                rel(CHART_DIR / "short_reversal_switch_equity_curve.svg"),
                rel(CHART_DIR / "short_reversal_switch_equity_curve.png"),
                rel(CHART_DIR / "short_reversal_switch_drawdown_curve.svg"),
                rel(CHART_DIR / "short_reversal_switch_drawdown_curve.png"),
                rel(CHART_DIR / "short_reversal_switch_last30_equity.svg"),
                rel(CHART_DIR / "short_reversal_switch_last30_equity.png"),
                rel(MONTE_DIR / "monte_pnl_fan_30d.svg"),
                rel(MONTE_DIR / "monte_pnl_fan_30d.png"),
                rel(MONTE_DIR / "monte_final_pnl_cdf_30d.svg"),
                rel(MONTE_DIR / "monte_final_pnl_cdf_30d.png"),
                rel(MONTE_DIR / "monte_maxdd_hist_30d.svg"),
                rel(MONTE_DIR / "monte_maxdd_hist_30d.png"),
                rel(MONTE_DIR / "monte_pnl_fan_100d.svg"),
                rel(MONTE_DIR / "monte_pnl_fan_100d.png"),
            ],
            "monte_carlo_metrics": rel(MODEL_DIR / "monte_carlo_metrics.json"),
            "supertrend_regime_audit": rel(MODEL_DIR / "supertrend_regime_audit.md"),
            "supertrend_filter_candidates": rel(MODEL_DIR / "supertrend_filter_candidates.csv"),
            "supertrend_variant_comparison": rel(MODEL_DIR / "supertrend_variant_comparison.md"),
            "supertrend_variant_comparison_csv": rel(MODEL_DIR / "supertrend_variant_comparison.csv"),
            "supertrend_regime_manifest": rel(DATA_DIR / "supertrend_regime_manifest.json"),
            "supertrend_variant_comparison_manifest": rel(
                DATA_DIR / "supertrend_variant_comparison_manifest.json"
            ),
            "short_reversal_switch_report": rel(MODEL_DIR / "short_reversal_switch_comparison.md"),
            "short_reversal_switch_summary": rel(MODEL_DIR / "short_reversal_switch_comparison.csv"),
            "short_reversal_switch_events": rel(MODEL_DIR / "short_reversal_switch_events.csv"),
            "short_reversal_switch_legs": rel(MODEL_DIR / "short_reversal_switch_legs.csv"),
            "short_reversal_switch_manifest": rel(
                DATA_DIR / "short_reversal_switch_comparison_manifest.json"
            ),
            "short_switch_tp2r_p0_report": rel(SHORT_SWITCH_P0_REPORT),
            "short_switch_tp2r_p0_summary": rel(SHORT_SWITCH_P0_CSV),
            "short_switch_tp2r_p0_best_events": rel(SHORT_SWITCH_P0_BEST_EVENTS),
            "short_switch_tp2r_p0_best_legs": rel(SHORT_SWITCH_P0_BEST_LEGS),
            "short_switch_tp2r_p0_manifest": rel(SHORT_SWITCH_P0_MANIFEST),
            "short_switch_tp2r_p0_full_report": rel(SHORT_SWITCH_P0_FULL_REPORT),
            "short_switch_tp2r_p0_best_yearly": rel(SHORT_SWITCH_P0_BEST_YEARLY),
            "short_switch_tp2r_p0_best_monthly": rel(SHORT_SWITCH_P0_BEST_MONTHLY),
            "data_control_artifacts": [
                rel(L0_1M_MANIFEST),
                rel(L0_1M_YF_MANIFEST),
                rel(L0_CONTINUITY_REPORT),
                rel(L0_5M_MANIFEST),
                rel(L0_15M_MANIFEST),
                rel(L0_PARITY_REPORT),
                rel(L0_DAILY_MANIFEST),
                rel(L1_CONTEXT_MANIFEST),
                rel(L1_AUDIT),
                rel(L1_DAILY_CONFLUENCE_MANIFEST),
                rel(L1_DAILY_CONFLUENCE_AUDIT),
                rel(SWEEP_MANIFEST),
                rel(PACKAGE_GATE),
            ],
        },
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
