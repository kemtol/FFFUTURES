#!/usr/bin/env python3
"""Compare MNQ ORB baseline against ST5_50 and long+short variants."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from build_supertrend_regime_features import (
    EVENTS_PATH,
    MODEL_DIR,
    SUMMARY_PATH,
    attach_features,
    build_st_table,
    load_anchor,
    load_context,
    load_events,
    max_drawdown,
    profit_factor,
)
from common import assert_mnq_namespaces, load_config, project_path, write_json

SWEEP_EVENTS_PATH = "data/Level_2_Datamart/mnq/ORB/sweeps/sweep_events.parquet"
OUTPUT_CSV = "supertrend_variant_comparison.csv"
OUTPUT_MD = "supertrend_variant_comparison.md"
RAW_BASE = (
    "https://raw.githubusercontent.com/kemtol/FFFUTURES/main/"
    "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
)
OUTPUT_JSON = (
    "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/"
    "supertrend_variant_comparison_manifest.json"
)
WINDOW_DAYS = [5, 10, 20, 30, 50, 100, 200]
CHART_STEMS = [
    "supertrend_variant_equity_curve",
    "supertrend_variant_drawdown_curve",
    "supertrend_variant_monthly_pnl_2026",
    "supertrend_variant_rolling_windows",
    "supertrend_variant_trade_pnl_distribution",
    "supertrend_variant_march_2026_equity",
]
VARIANT_COLORS = {
    "long_only_no_st": "#334155",
    "long_only_st5_50": "#0f766e",
    "long_short_no_st": "#dc2626",
    "long_short_st5_50_aligned": "#2563eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--st-timeframe", type=int, default=5)
    parser.add_argument("--st-period", type=int, default=50)
    parser.add_argument("--orb-minutes", type=int, default=15)
    parser.add_argument("--target-risk", type=float, default=500.0)
    parser.add_argument("--exit-mode", default="tp_2r_or_time")
    return parser.parse_args()


def raw(path: str) -> str:
    return f"{RAW_BASE}/{path}"


def clean_svg(path: Path) -> None:
    text = path.read_text()
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def save_fig(fig, chart_dir: Path, stem: str) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    svg_path = chart_dir / f"{stem}.svg"
    png_path = chart_dir / f"{stem}.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=150)
    clean_svg(svg_path)


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def prepare_event_frame(df: pd.DataFrame, pnl_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        out[col] = pd.to_datetime(out[col], utc=True)
    out["pnl_net_usd"] = out[pnl_col].astype(float)
    return out.sort_values("signal_ts").reset_index(drop=True)


def load_sweep_long_short(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(project_path(SWEEP_EVENTS_PATH))
    mask = (
        df["orb_minutes"].eq(args.orb_minutes)
        & df["side_mode"].eq("long_short")
        & df["exit_mode"].eq(args.exit_mode)
        & df["target_risk_usd"].astype(float).eq(float(args.target_risk))
    )
    out = df[mask].copy()
    if out.empty:
        raise SystemExit("No long_short sweep events found for requested variant.")
    return prepare_event_frame(out, "pnl_usd")


def subset_dates(events: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(events["ny_date"])
    return events[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def summarize(events: pd.DataFrame, anchor: pd.Timestamp) -> dict[str, Any]:
    pnl = events["pnl_net_usd"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total = float(pnl.sum()) if not pnl.empty else 0.0
    dd = max_drawdown(pnl)
    out: dict[str, Any] = {
        "trades": int(len(events)),
        "long_trades": int(events["side"].eq("LONG").sum()) if "side" in events else int(len(events)),
        "short_trades": int(events["side"].eq("SHORT").sum()) if "side" in events else 0,
        "win_rate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
        "pnl_usd": total,
        "max_dd_usd": dd,
        "return_dd": (total / abs(dd)) if dd else None,
        "profit_factor": profit_factor(pnl),
        "avg_trade_usd": float(pnl.mean()) if not pnl.empty else 0.0,
        "gross_profit_usd": float(wins.sum()) if not wins.empty else 0.0,
        "gross_loss_usd": float(losses.sum()) if not losses.empty else 0.0,
    }
    jan_may = subset_dates(events, "2026-01-01", "2026-05-31")
    march = subset_dates(events, "2026-03-01", "2026-03-31")
    for prefix, subset in [("jan_may_2026", jan_may), ("march_2026", march)]:
        sub_pnl = subset["pnl_net_usd"].astype(float)
        out[f"{prefix}_trades"] = int(len(subset))
        out[f"{prefix}_pnl_usd"] = float(sub_pnl.sum()) if not sub_pnl.empty else 0.0
        out[f"{prefix}_max_dd_usd"] = max_drawdown(sub_pnl) if not sub_pnl.empty else 0.0
        out[f"{prefix}_win_rate"] = float((sub_pnl > 0).mean()) if not sub_pnl.empty else 0.0
    for days in WINDOW_DAYS:
        window = events[
            (events["signal_ts"] > anchor - pd.Timedelta(days=days))
            & (events["signal_ts"] <= anchor)
        ]
        w_pnl = window["pnl_net_usd"].astype(float)
        out[f"last_{days}d_trades"] = int(len(window))
        out[f"last_{days}d_pnl_usd"] = float(w_pnl.sum()) if not w_pnl.empty else 0.0
        out[f"last_{days}d_max_dd_usd"] = max_drawdown(w_pnl) if not w_pnl.empty else 0.0
    return out


def build_variant_frames(
    baseline: pd.DataFrame,
    long_short: pd.DataFrame,
    st_feature: str,
) -> list[tuple[str, str, str, pd.DataFrame]]:
    long_st = baseline[baseline[f"{st_feature}_bullish"].fillna(False)].copy()
    aligned = (
        (long_short["side"].eq("LONG") & long_short[f"{st_feature}_bullish"].fillna(False))
        | (long_short["side"].eq("SHORT") & long_short[f"{st_feature}_bearish"].fillna(False))
    )
    long_short_st = long_short[aligned].copy()

    variants = [
        (
            "long_only_no_st",
            "Long only, no ST",
            "Baseline rule: first M1 close above OR high.",
            baseline,
        ),
        (
            "long_only_st5_50",
            "Long only, ST5_50 bullish",
            "Take long breakout only when ST5_50 is bullish/up at signal close.",
            long_st,
        ),
        (
            "long_short_no_st",
            "Long+Short, no ST",
            "First M1 breakout above OR high or below OR low; direction follows breakout.",
            long_short,
        ),
        (
            "long_short_st5_50_aligned",
            "Long+Short, ST5_50 aligned",
            "Long requires bullish ST5_50; short requires bearish ST5_50 at signal close.",
            long_short_st,
        ),
    ]
    prepared = []
    for variant_id, label, rule, frame in variants:
        cur = frame.sort_values("signal_ts").reset_index(drop=True).copy()
        cur["variant_id"] = variant_id
        cur["variant_label"] = label
        cur["variant_rule"] = rule
        prepared.append((variant_id, label, rule, cur))
    return prepared


def build_rows(variants: list[tuple[str, str, str, pd.DataFrame]], anchor: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for variant_id, label, rule, frame in variants:
        row = {
            "variant_id": variant_id,
            "label": label,
            "rule": rule,
            "anchor_ts": anchor.isoformat(),
        }
        row.update(summarize(frame, anchor))
        rows.append(row)
    out = pd.DataFrame(rows)
    baseline_row = out[out["variant_id"].eq("long_only_no_st")].iloc[0]
    out["pnl_delta_vs_baseline"] = out["pnl_usd"] - baseline_row["pnl_usd"]
    out["dd_improvement_vs_baseline"] = out["max_dd_usd"] - baseline_row["max_dd_usd"]
    out["march_pnl_delta_vs_baseline"] = (
        out["march_2026_pnl_usd"] - baseline_row["march_2026_pnl_usd"]
    )
    out["march_dd_improvement_vs_baseline"] = (
        out["march_2026_max_dd_usd"] - baseline_row["march_2026_max_dd_usd"]
    )
    return out


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${float(value):,.0f}"


def num(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def plot_equity_curve(variants: list[tuple[str, str, str, pd.DataFrame]], chart_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for variant_id, label, _, frame in variants:
        equity = frame["pnl_net_usd"].astype(float).cumsum()
        ax.plot(
            frame["signal_ts"],
            equity,
            label=label,
            linewidth=1.8,
            color=VARIANT_COLORS.get(variant_id),
        )
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "MNQ ORB ST5_50 Variant Equity Curve", "Cumulative PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, chart_dir, "supertrend_variant_equity_curve")
    plt.close(fig)


def plot_drawdown_curve(variants: list[tuple[str, str, str, pd.DataFrame]], chart_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for variant_id, label, _, frame in variants:
        equity = frame["pnl_net_usd"].astype(float).cumsum()
        drawdown = equity - equity.cummax()
        ax.plot(
            frame["signal_ts"],
            drawdown,
            label=label,
            linewidth=1.5,
            color=VARIANT_COLORS.get(variant_id),
        )
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "MNQ ORB ST5_50 Variant Drawdown", "Drawdown ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, chart_dir, "supertrend_variant_drawdown_curve")
    plt.close(fig)


def plot_monthly_2026(variants: list[tuple[str, str, str, pd.DataFrame]], chart_dir: Path) -> None:
    months = pd.period_range("2026-01", "2026-05", freq="M").astype(str).tolist()
    labels = [label for _, label, _, _ in variants]
    x = list(range(len(months)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for i, (variant_id, label, _, frame) in enumerate(variants):
        monthly = (
            frame.assign(month=pd.to_datetime(frame["ny_date"]).dt.to_period("M").astype(str))
            .groupby("month")["pnl_net_usd"]
            .sum()
            .reindex(months, fill_value=0.0)
        )
        offsets = [v + (i - 1.5) * width for v in x]
        ax.bar(
            offsets,
            monthly.values,
            width=width,
            label=label,
            color=VARIANT_COLORS.get(variant_id),
            alpha=0.88,
        )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    style_axis(ax, "MNQ ORB ST5_50 Variant Monthly PnL 2026", "Monthly PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, chart_dir, "supertrend_variant_monthly_pnl_2026")
    plt.close(fig)


def plot_rolling_windows(comparison: pd.DataFrame, chart_dir: Path) -> None:
    windows = [5, 10, 20, 30, 50, 100, 200]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    for _, row in comparison.iterrows():
        variant_id = row["variant_id"]
        label = row["label"]
        pnls = [row[f"last_{days}d_pnl_usd"] for days in windows]
        dds = [row[f"last_{days}d_max_dd_usd"] for days in windows]
        axes[0].plot(windows, pnls, marker="o", label=label, color=VARIANT_COLORS.get(variant_id))
        axes[1].plot(windows, dds, marker="o", label=label, color=VARIANT_COLORS.get(variant_id))
    axes[0].axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    axes[1].axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(axes[0], "Rolling Window PnL", "PnL ($)")
    style_axis(axes[1], "Rolling Window Max Drawdown", "Drawdown ($)")
    axes[1].set_xlabel("Lookback days")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, chart_dir, "supertrend_variant_rolling_windows")
    plt.close(fig)


def plot_trade_distribution(variants: list[tuple[str, str, str, pd.DataFrame]], chart_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for variant_id, label, _, frame in variants:
        ax.hist(
            frame["pnl_net_usd"].astype(float),
            bins=55,
            histtype="step",
            linewidth=1.5,
            label=label,
            color=VARIANT_COLORS.get(variant_id),
        )
    ax.axvline(0, color="#334155", linewidth=0.8, alpha=0.6)
    style_axis(ax, "MNQ ORB ST5_50 Variant Trade PnL Distribution", "Trade count")
    ax.set_xlabel("Trade PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, chart_dir, "supertrend_variant_trade_pnl_distribution")
    plt.close(fig)


def plot_march_2026_equity(variants: list[tuple[str, str, str, pd.DataFrame]], chart_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for variant_id, label, _, frame in variants:
        dates = pd.to_datetime(frame["ny_date"])
        march = frame[(dates >= pd.Timestamp("2026-03-01")) & (dates <= pd.Timestamp("2026-03-31"))]
        if march.empty:
            continue
        equity = march["pnl_net_usd"].astype(float).cumsum()
        ax.plot(
            march["signal_ts"],
            equity,
            marker="o",
            label=label,
            linewidth=1.5,
            color=VARIANT_COLORS.get(variant_id),
        )
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    style_axis(ax, "MNQ ORB ST5_50 Variant March 2026 Equity", "March cumulative PnL ($)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, chart_dir, "supertrend_variant_march_2026_equity")
    plt.close(fig)


def save_variant_charts(
    variants: list[tuple[str, str, str, pd.DataFrame]],
    comparison: pd.DataFrame,
    chart_dir: Path,
) -> list[str]:
    plot_equity_curve(variants, chart_dir)
    plot_drawdown_curve(variants, chart_dir)
    plot_monthly_2026(variants, chart_dir)
    plot_rolling_windows(comparison, chart_dir)
    plot_trade_distribution(variants, chart_dir)
    plot_march_2026_equity(variants, chart_dir)
    charts = []
    for stem in CHART_STEMS:
        charts.append(str((chart_dir / f"{stem}.svg").relative_to(project_path("."))))
        charts.append(str((chart_dir / f"{stem}.png").relative_to(project_path("."))))
    return charts


def write_report(path: Path, df: pd.DataFrame, manifest: dict[str, Any]) -> None:
    cols = [
        ("label", "Variant"),
        ("trades", "Trades"),
        ("long_trades", "Long"),
        ("short_trades", "Short"),
        ("win_rate", "WR"),
        ("pnl_usd", "PnL"),
        ("max_dd_usd", "DD"),
        ("return_dd", "Ret/DD"),
        ("jan_may_2026_trades", "Jan-May Trades"),
        ("jan_may_2026_pnl_usd", "Jan-May PnL"),
        ("jan_may_2026_max_dd_usd", "Jan-May DD"),
        ("march_2026_pnl_usd", "Mar PnL"),
        ("march_2026_max_dd_usd", "Mar DD"),
        ("last_30d_trades", "30D Trades"),
        ("last_30d_pnl_usd", "30D PnL"),
        ("last_30d_max_dd_usd", "30D DD"),
    ]
    lines = [
        "# MNQ ORB ST5_50 Variant Comparison",
        "",
        "## Scope",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| ORB | 15m New York opening range |",
        "| Exit | TP 2R or 15:00 NY EOD/time exit |",
        "| Risk | $500 target risk |",
        "| ST variant | ST5_50, factor 4.0 |",
        "| Long ST rule | `ST5_50_dir == -1` bullish/up |",
        "| Short ST rule | `ST5_50_dir == +1` bearish/down |",
        "| Join rule | Latest completed ST timestamp `<= signal_ts` |",
        f"| Anchor | {manifest['anchor_ts']} |",
        f"| Lookahead violations | {manifest['lookahead_violations']} |",
        "",
        "## Visual Artifacts",
        "",
        "### Equity Curve",
        "",
        f"![ST5_50 Variant Equity Curve]({raw('charts/supertrend_variant_equity_curve.png')})",
        "",
        "### Drawdown Curve",
        "",
        f"![ST5_50 Variant Drawdown Curve]({raw('charts/supertrend_variant_drawdown_curve.png')})",
        "",
        "### Monthly PnL 2026",
        "",
        f"![ST5_50 Variant Monthly PnL 2026]({raw('charts/supertrend_variant_monthly_pnl_2026.png')})",
        "",
        "### Rolling Window PnL/DD",
        "",
        f"![ST5_50 Variant Rolling Windows]({raw('charts/supertrend_variant_rolling_windows.png')})",
        "",
        "### Trade PnL Distribution",
        "",
        f"![ST5_50 Variant Trade PnL Distribution]({raw('charts/supertrend_variant_trade_pnl_distribution.png')})",
        "",
        "### March 2026 Equity",
        "",
        f"![ST5_50 Variant March 2026 Equity]({raw('charts/supertrend_variant_march_2026_equity.png')})",
        "",
        "## Comparison",
        "",
        "| " + " | ".join(label for _, label in cols) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i, _ in enumerate(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col, _ in cols:
            if col.endswith("pnl_usd") or col.endswith("dd_usd"):
                values.append(money(row[col]))
            elif col == "win_rate":
                values.append(pct(row[col]))
            elif col == "return_dd":
                values.append(num(row[col]))
            elif col.endswith("trades") or col in {"trades", "long_trades", "short_trades"}:
                values.append(str(int(row[col])))
            else:
                values.append(str(row[col]))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Current Read",
            "",
            "- `ST5_50` as long-only filter is the clean P0 candidate because it fixes March with only one extra rule.",
            "- `Long+Short no ST` adds frequency, but the short leg is weak by itself.",
            "- `Long+Short ST5_50 aligned` is worth tracking only if it improves drawdown without diluting the long-only edge.",
            "",
            "## Artifacts",
            "",
            "| Artifact | Path |",
            "| --- | --- |",
            f"| CSV | `{manifest['artifacts']['csv']}` |",
            f"| Charts | `{manifest['artifacts']['charts_dir']}` |",
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

    model_dir = project_path(MODEL_DIR)
    chart_dir = model_dir / "charts"
    csv_path = model_dir / OUTPUT_CSV
    md_path = model_dir / OUTPUT_MD
    manifest_path = project_path(OUTPUT_JSON)
    for path in [csv_path, md_path, manifest_path]:
        if path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing artifact without --force: {path}")

    baseline = prepare_event_frame(load_events(project_path(EVENTS_PATH)), "pnl_net_usd")
    anchor = load_anchor(baseline)
    long_short = load_sweep_long_short(args)

    m1 = load_context(project_path(cfg["outputs"]["l1_context"]))
    table, table_audit = build_st_table(
        m1,
        timeframe_minutes=args.st_timeframe,
        atr_periods=[args.st_period],
        factor=args.factor,
    )
    feature_tables = {args.st_timeframe: table}
    baseline, baseline_audit = attach_features(baseline, feature_tables, [args.st_period])
    long_short, long_short_audit = attach_features(long_short, feature_tables, [args.st_period])
    st_feature = f"st{args.st_timeframe}_{args.st_period}"
    variants = build_variant_frames(baseline, long_short, st_feature)
    comparison = build_rows(variants, anchor)

    model_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)
    chart_outputs = save_variant_charts(variants, comparison, chart_dir)
    lookahead_violations = int(
        sum(baseline_audit["lookahead_violations"].values())
        + sum(long_short_audit["lookahead_violations"].values())
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchor_ts": anchor.isoformat(),
        "params": {
            "orb_minutes": args.orb_minutes,
            "target_risk_usd": args.target_risk,
            "exit_mode": args.exit_mode,
            "st_timeframe": args.st_timeframe,
            "st_period": args.st_period,
            "st_factor": args.factor,
        },
        "sources": {
            "baseline_events": EVENTS_PATH,
            "long_short_events": SWEEP_EVENTS_PATH,
            "l1_context": cfg["outputs"]["l1_context"],
        },
        "feature_audit": table_audit,
        "baseline_attach_audit": baseline_audit,
        "long_short_attach_audit": long_short_audit,
        "lookahead_violations": lookahead_violations,
        "artifacts": {
            "csv": str(csv_path.relative_to(project_path("."))),
            "report": str(md_path.relative_to(project_path("."))),
            "charts_dir": str(chart_dir.relative_to(project_path("."))),
            "charts": chart_outputs,
            "manifest": OUTPUT_JSON,
        },
    }
    write_json(manifest_path, manifest)
    write_report(md_path, comparison, manifest)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    print(
        comparison[
            [
                "label",
                "trades",
                "long_trades",
                "short_trades",
                "pnl_usd",
                "max_dd_usd",
                "return_dd",
                "jan_may_2026_pnl_usd",
                "march_2026_pnl_usd",
                "march_2026_max_dd_usd",
                "last_30d_pnl_usd",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
