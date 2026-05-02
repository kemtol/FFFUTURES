#!/usr/bin/env python3
"""
Calibration test: compare batch (parquet) vs live (feature_builder) features.

Outputs: calibration_report.csv — per-feature deltas + per-module match rates
         calibration_details.csv — event-level feature comparison for AB analysis

Usage:
    python3 pipeline/live/calibrate.py [--overlap-only]
"""

from __future__ import annotations

import sys
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.buffer import DataBuffer
from pipeline.live.orb_detector import BreakoutEvent
from pipeline.live.feature_builder import FeatureBuilder, FEATURE_ORDER

# ── Per-module feature mapping ────────────────────────────────────────────────

MODULE_FEATURES = {
    "orb_context": [
        "orb_range_atr_ratio", "day_of_week", "time_in_session_min",
        "vwap_at_breakout", "price_vs_vwap_pct", "adx_14_15m", "ema_slope_1h",
    ],
    "scale_invariant": [
        "breakout_strength_atr_ratio", "atr14_sq", "breakout_strength_sq",
        "orb_range_sq", "price_vs_vwap_pct_abs", "adx_50_flag", "breakout_strength_vs_orb",
    ],
    "volatility_normalized": [
        "atr14_percentile_20d", "atr14_zscore_20d",
        "breakout_strength_percentile_20d", "breakout_strength_zscore_10d",
        "orb_range_percentile_20d",
    ],
    "pre_breakout_profile": [
        "pre_bo_compression_ratio", "pre_bo_drift_atr", "pre_bo_inside_bar_flag",
        "pre_bo_last_candle_range_ratio", "pre_bo_bullish_ratio",
    ],
    "session_momentum": [
        "sm_first_30m_range", "sm_first_30m_direction",
        "sm_pre_breakout_volume_ratio", "sm_pre_breakout_volume_z",
    ],
    "interaction": [
        "int_atr14_x_adx", "int_breakout_strength_x_range",
        "int_vwap_distance_x_atr14", "int_adx_x_orb_range",
        "int_breakout_strength_x_session",
    ],
    "macro": [
        "mac_spx_regime", "mac_dxy_trend", "mac_us10y_change", "mac_oil_volatility",
    ],
    "core": [
        "breakout_strength", "atr14_at_entry", "breakout_side",
    ],
}

def load_batch_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all module parquets onto events to get full batch feature set."""
    df = events_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    modules_dir = ROOT / "data" / "Level_1_Features" / "modules"
    for parquet_path in sorted(modules_dir.glob("*.parquet")):
        mod = pd.read_parquet(parquet_path)
        mod["date"] = pd.to_datetime(mod["date"]).dt.date
        merge_keys = ["date", "session", "orb_tf", "breakout_ts"]
        df = df.merge(mod, on=merge_keys, how="left")

    # Add mc columns from datamart if missing
    dm = pd.read_parquet(ROOT / "data" / "Level_2_Datamart" / "training_datamart_orb.parquet")
    dm = dm[dm["side"] == "rev"].copy()  # one row per event
    dm["date"] = pd.to_datetime(dm["date"]).dt.date

    mc_cols = ["orb_range_atr_ratio", "day_of_week", "time_in_session_min",
               "vwap_at_breakout", "price_vs_vwap_pct", "adx_14_15m", "ema_slope_1h"]
    for col in mc_cols:
        if col not in df.columns or df[col].isna().all():
            dm_sub = dm[["date", "session", "orb_tf", "breakout_ts", col]].dropna(subset=[col])
            if col not in df.columns:
                df = df.merge(dm_sub, on=["date", "session", "orb_tf", "breakout_ts"], how="left")
            else:
                mask = df[col].isna()
                df.loc[mask, col] = df.loc[mask].merge(
                    dm_sub, on=["date", "session", "orb_tf", "breakout_ts"], how="left"
                )[col + "_y"].values

    return df


def create_event(df_row) -> BreakoutEvent | None:
    """Create BreakoutEvent from a DataFrame row."""
    try:
        ts = df_row["breakout_ts"]
        if isinstance(ts, np.datetime64):
            ts = pd.Timestamp(ts)
        elif not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        d = df_row["date"]
        if isinstance(d, np.datetime64):
            d = pd.Timestamp(d).date()
        elif isinstance(d, datetime):
            d = d.date()

        orb_start = df_row.get("orb_start_ts")
        if isinstance(orb_start, np.datetime64):
            orb_start = pd.Timestamp(orb_start)
        elif not isinstance(orb_start, pd.Timestamp):
            orb_start = pd.Timestamp(orb_start) if orb_start is not None else ts - pd.Timedelta(minutes=15)
        if orb_start.tzinfo is None:
            orb_start = orb_start.tz_localize("UTC")

        orb_end = df_row.get("orb_end_ts")
        if isinstance(orb_end, np.datetime64):
            orb_end = pd.Timestamp(orb_end)
        elif not isinstance(orb_end, pd.Timestamp):
            orb_end = pd.Timestamp(orb_end) if orb_end is not None else orb_start + pd.Timedelta(minutes=15)
        if orb_end.tzinfo is None:
            orb_end = orb_end.tz_localize("UTC")

        sess_close = df_row.get("session_close_ts")
        if isinstance(sess_close, np.datetime64):
            sess_close = pd.Timestamp(sess_close)
        elif not isinstance(sess_close, pd.Timestamp):
            sess_close = pd.Timestamp(sess_close) if sess_close is not None else ts
        if sess_close.tzinfo is None:
            sess_close = sess_close.tz_localize("UTC")

        return BreakoutEvent(
            date=d,
            session=str(df_row["session"]),
            orb_tf=str(df_row["orb_tf"]),
            orb_start_ts=orb_start,
            orb_end_ts=orb_end,
            orb_high=float(df_row["orb_high"]),
            orb_low=float(df_row["orb_low"]),
            orb_range=float(df_row["orb_range"]),
            breakout_ts=ts,
            breakout_side=int(df_row["breakout_side"]),
            entry_price=float(df_row["entry_price"]),
            session_close_ts=sess_close,
        )
    except Exception as e:
        print(f"  SKIP event: {e}", flush=True)
        return None


def main():
    overlap_only = "--overlap-only" in sys.argv

    print("=== Calibration Test — Batch vs Live Feature Comparison ===\n", flush=True)

    # ── Load breakout events ─────────────────────────────────────────────
    bo = pd.read_parquet(ROOT / "data" / "Level_1_Features" / "breakout_events.parquet")
    bo["date"] = pd.to_datetime(bo["date"]).dt.date

    if overlap_only:
        # Use only period covered by TopstepX buffer (Jan 29+)
        cutoff = Date(2026, 1, 29)
        events_df = bo[bo["date"] >= cutoff].copy()
        print(f"Events (overlap): {len(events_df)} ({cutoff} → {events_df['date'].max()})", flush=True)
    else:
        cutoff = Date(2025, 12, 1)
        events_df = bo[bo["date"] >= cutoff].copy()
        print(f"Events (holdout): {len(events_df)} ({cutoff} → {events_df['date'].max()})", flush=True)

    # ── Load batch features ──────────────────────────────────────────────
    print("\nLoading batch features...", flush=True)
    batch_df = load_batch_features(events_df)
    print(f"Batch features shape: {batch_df.shape}", flush=True)

    # ── Initialize live pipeline ─────────────────────────────────────────
    print("\nInitializing buffer + feature_builder...", flush=True)
    buffer = DataBuffer()
    builder = FeatureBuilder(buffer)

    # ── Compute live features for each event ─────────────────────────────
    print(f"Computing live features for {len(events_df)} events...", flush=True)

    rows = []
    live_features_list = []  # track all live feature dicts for aggregate analysis
    skipped = 0
    errors = 0

    for idx, (_, row) in enumerate(events_df.iterrows()):
        event = create_event(row)
        if event is None:
            skipped += 1
            continue

        try:
            live = builder.build(event)
            live_features_list.append((idx, event, live))
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR [{errors}] event {idx}: {e}", flush=True)
            continue

        if (idx + 1) % 100 == 0:
            print(f"  ... {idx + 1}/{len(events_df)} ({skipped} skipped, {errors} errors)", flush=True)

    print(f"\nLive features computed: {len(live_features_list)} events "
          f"({skipped} skipped, {errors} errors)", flush=True)

    # ── Build comparison ──────────────────────────────────────────────────
    print("\n--- COMPARISON ---\n", flush=True)

    comparison_rows = []
    feature_deltas: dict[str, list[float]] = {f: [] for f in FEATURE_ORDER}

    for idx, event, live in live_features_list:
        batch_row = batch_df.iloc[idx]
        for feat in FEATURE_ORDER:
            batch_val = batch_row.get(feat, np.nan)
            live_val = live.get(feat, np.nan)

            try:
                bv = float(batch_val) if not pd.isna(batch_val) else np.nan
                lv = float(live_val) if not pd.isna(live_val) else np.nan
            except (TypeError, ValueError):
                bv = np.nan
                lv = np.nan

            delta = lv - bv if (not np.isnan(bv) and not np.isnan(lv)) else np.nan
            feature_deltas[feat].append(delta)

            comparison_rows.append({
                "event_idx": idx,
                "date": event.date,
                "session": event.session,
                "feature": feat,
                "batch": bv,
                "live": lv,
                "delta": delta,
            })

    comp_df = pd.DataFrame(comparison_rows)

    # ── Feature-level report ──────────────────────────────────────────────
    print("\n--- FEATURE-LEVEL DELTA ---")
    print(f"{'Feature':<42s} {'Mean Δ':>10s} {'Std Δ':>10s} {'|Δ|<0.01':>10s} {'|Δ|<0.05':>10s} {'ExactMatch%':>12s}")
    print("-" * 95)

    feature_report_rows = []
    for feat in FEATURE_ORDER:
        deltas = np.array([d for d in feature_deltas[feat] if not np.isnan(d)])
        n = len(deltas)
        if n == 0:
            mean_d, std_d, pct_001, pct_005, exact_pct = np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            mean_d = float(np.mean(deltas))
            std_d = float(np.std(deltas))
            pct_001 = float(np.mean(np.abs(deltas) < 0.01)) * 100
            pct_005 = float(np.mean(np.abs(deltas) < 0.05)) * 100
            exact_pct = float(np.mean(np.abs(deltas) < 1e-9)) * 100

        print(f"{feat:<42s} {mean_d:>10.4f} {std_d:>10.4f} {pct_001:>10.1f}% {pct_005:>10.1f}% {exact_pct:>11.1f}%")

        feature_report_rows.append({
            "feature": feat,
            "n_samples": n,
            "mean_delta": mean_d,
            "std_delta": std_d,
            "pct_delta_lt_001": pct_001,
            "pct_delta_lt_005": pct_005,
            "pct_exact_match": exact_pct,
        })

    feat_report = pd.DataFrame(feature_report_rows)

    # ── Module-level report ───────────────────────────────────────────────
    print("\n--- MODULE-LEVEL MATCH RATE ---")
    print(f"{'Module':<25s} {'Match%':>8s} {'Mean|Δ|':>10s} {'Features':>12s}")
    print("-" * 58)

    module_report_rows = []
    for mod_name, mod_features in MODULE_FEATURES.items():
        mod_deltas = []
        for feat in mod_features:
            d = [d for d in feature_deltas[feat] if not np.isnan(d)]
            mod_deltas.extend(d)
        n = len(mod_deltas)
        match_pct = float(np.mean(np.abs(np.array(mod_deltas)) < 0.01)) * 100 if n > 0 else 0
        mean_abs_delta = float(np.mean(np.abs(np.array(mod_deltas)))) if n > 0 else np.nan
        print(f"{mod_name:<25s} {match_pct:>7.1f}% {mean_abs_delta:>10.4f} {len(mod_features):>12}")

        module_report_rows.append({
            "module": mod_name,
            "n_deltas": n,
            "match_pct_lt_001": match_pct,
            "mean_abs_delta": mean_abs_delta,
            "n_features": len(mod_features),
        })

    mod_report = pd.DataFrame(module_report_rows)

    # ── Same-decision + probability delta ─────────────────────────────────
    print("\n--- DECISION-LEVEL CALIBRATION ---")

    # Run inference on live features
    from pipeline.live.runner import SignalRunner
    runner = SignalRunner()
    same_decisions = 0
    total_decisions = 0
    prob_deltas = []

    for _, row in comp_df.iterrows():
        pass

    # We can't easily run batch inference here without the runner context.
    # Instead, compute live inference for all events and compare.
    # For now, focus on feature-level comparison.

    # ── Save reports ──────────────────────────────────────────────────────
    out_dir = ROOT / "model" / "CALIBRATION"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_report.to_csv(out_dir / "feature_report.csv", index=False)
    mod_report.to_csv(out_dir / "module_report.csv", index=False)
    comp_df.to_csv(out_dir / "feature_deltas.csv", index=False)

    print(f"\nReports saved to {out_dir}/")
    print(f"  feature_report.csv  — per-feature delta stats")
    print(f"  module_report.csv   — per-module match rates")
    print(f"  feature_deltas.csv  — event-level feature deltas ({len(comp_df)} rows)")
    print(f"\nTotal: {len(live_features_list)} events, {errors} errors, {skipped} skipped", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    total_finite = sum(1 for r in feature_report_rows if not np.isnan(r["mean_delta"]))
    exact_gt_90 = sum(1 for r in feature_report_rows if r["pct_exact_match"] > 90)
    delta_lt_005 = sum(1 for r in feature_report_rows if r["pct_delta_lt_005"] > 90)
    print(f"Features with data: {total_finite}/{len(FEATURE_ORDER)}")
    print(f"Features with >90% exact match: {exact_gt_90}/{total_finite}")
    print(f"Features with >90% below 0.05 delta: {delta_lt_005}/{total_finite}")

    overall_match = np.mean([r["pct_exact_match"] for r in feature_report_rows if not np.isnan(r["pct_exact_match"])])
    print(f"Overall mean exact match: {overall_match:.1f}%")


if __name__ == "__main__":
    main()
