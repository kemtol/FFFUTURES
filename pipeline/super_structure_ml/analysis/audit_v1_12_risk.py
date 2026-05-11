#!/usr/bin/env python3
"""
Risk bucket audit for SMART_1 aggressive v1.12.

This is a quick-win diagnostic for the mechanical pullback baseline. It does
not train a model and does not mutate the datamart. It asks whether a simple
`risk_pts` cap can preserve recent-regime PnL while reducing drawdown.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/kemal/futures")
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet"
REPORT_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1/reports"


def max_drawdown(pnls: pd.Series) -> float:
    balance = 50_000.0
    peak = 50_000.0
    max_dd = 0.0
    for pnl in pnls.astype(float):
        balance += pnl
        peak = max(peak, balance)
        max_dd = min(max_dd, balance - peak)
    return float(max_dd)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_pnl": np.nan,
            "total_pnl": 0.0,
            "max_dd": 0.0,
            "avg_risk_pts": np.nan,
        }
    ordered = df.sort_values("entry_ts")
    return {
        "trades": int(len(ordered)),
        "win_rate": float((ordered["pnl_usd"] > 0).mean()),
        "avg_pnl": float(ordered["pnl_usd"].mean()),
        "total_pnl": float(ordered["pnl_usd"].sum()),
        "max_dd": max_drawdown(ordered["pnl_usd"]),
        "avg_risk_pts": float(ordered["risk_pts"].mean()),
    }


def add_windows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    latest = df["entry_ts"].max()
    return {
        "all": df,
        "2026": df[df["entry_ts"].dt.year == 2026],
        "latest_30d": df[df["entry_ts"] >= latest - pd.Timedelta(days=30)],
        "apr_may_2026": df[
            (df["entry_ts"] >= pd.Timestamp("2026-04-01", tz="UTC"))
            & (df["entry_ts"] < pd.Timestamp("2026-06-01", tz="UTC"))
        ],
        "oot_200d": df[df["entry_ts"] >= latest - pd.Timedelta(days=200)],
    }


def build_bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, np.inf]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20", "20-30", "30-50", "50+"]
    work = df.copy()
    work["risk_bucket"] = pd.cut(work["risk_pts"], bins=edges, labels=labels, right=False)

    rows = []
    for window_name, window_df in add_windows(work).items():
        for bucket, bucket_df in window_df.groupby("risk_bucket", observed=False):
            row = {"window": window_name, "risk_bucket": str(bucket)}
            row.update(summarize(bucket_df))
            rows.append(row)
    return pd.DataFrame(rows)


def build_cutoff_table(df: pd.DataFrame) -> pd.DataFrame:
    quantile_cutoffs = df["risk_pts"].quantile([0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]).tolist()
    fixed_cutoffs = [4, 6, 8, 10, 12, 15, 20, 25, 30]
    cutoffs = sorted({round(float(x), 2) for x in fixed_cutoffs + quantile_cutoffs if x > 0})

    rows = []
    windows = add_windows(df)
    for cutoff in cutoffs:
        for window_name, window_df in windows.items():
            kept = window_df[window_df["risk_pts"] <= cutoff]
            dropped = len(window_df) - len(kept)
            row = {
                "risk_cap_pts": cutoff,
                "window": window_name,
                "dropped_trades": int(dropped),
                "drop_rate": float(dropped / len(window_df)) if len(window_df) else np.nan,
            }
            row.update(summarize(kept))
            rows.append(row)
    return pd.DataFrame(rows)


def print_window(title: str, df: pd.DataFrame) -> None:
    s = summarize(df)
    print(
        f"{title:14s} trades={s['trades']:4d} "
        f"pnl=${s['total_pnl']:9.2f} avg=${s['avg_pnl']:7.2f} "
        f"win={s['win_rate']:.3f} mdd=${s['max_dd']:9.2f} "
        f"avg_risk={s['avg_risk_pts']:.2f}"
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATAMART_PATH)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df = df.sort_values("entry_ts").reset_index(drop=True)

    print(f"Loaded {len(df)} rows from {DATAMART_PATH}")
    print(f"Date range: {df['entry_ts'].min()} -> {df['entry_ts'].max()}")
    print("\nBase windows")
    for name, window_df in add_windows(df).items():
        print_window(name, window_df)

    bucket_table = build_bucket_table(df)
    cutoff_table = build_cutoff_table(df)

    bucket_path = REPORT_DIR / "v1_12_risk_bucket_audit.csv"
    cutoff_path = REPORT_DIR / "v1_12_risk_cutoff_audit.csv"
    bucket_table.to_csv(bucket_path, index=False)
    cutoff_table.to_csv(cutoff_path, index=False)

    print(f"\nSaved bucket audit: {bucket_path}")
    print(f"Saved cutoff audit: {cutoff_path}")

    print("\nLatest 30d risk caps")
    latest = cutoff_table[cutoff_table["window"] == "latest_30d"].sort_values(
        ["total_pnl", "max_dd"], ascending=[False, False]
    )
    cols = ["risk_cap_pts", "trades", "dropped_trades", "total_pnl", "avg_pnl", "win_rate", "max_dd"]
    print(latest[cols].head(12).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n2026 risk caps")
    recent = cutoff_table[cutoff_table["window"] == "2026"].sort_values(
        ["total_pnl", "max_dd"], ascending=[False, False]
    )
    print(recent[cols].head(12).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
