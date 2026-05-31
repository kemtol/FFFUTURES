#!/usr/bin/env python3
"""Simulate fractional Kelly risk overlays for MNQ ORB breakout quality."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from common import project_path, write_json  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_MODEL_ARTIFACT = "model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_success_2r_logistic.joblib"
DEFAULT_EXECUTION_EVENTS = "data/Level_2_Datamart/mnq/orb_vol_target/sweeps/sweep_base_opportunities.parquet"
DEFAULT_KELLY_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
DEFAULT_NORMALIZED_TARGET_RISKS = [600.0, 750.0, 1000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--model-artifact", default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument("--execution-events", default=DEFAULT_EXECUTION_EVENTS)
    parser.add_argument("--base-risk-usd", type=float, default=500.0)
    parser.add_argument("--payoff-ratio", type=float, default=2.0)
    parser.add_argument("--min-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--max-risk-multiplier", type=float, default=2.0)
    parser.add_argument("--kelly-fractions", default=",".join(str(x) for x in DEFAULT_KELLY_FRACTIONS))
    parser.add_argument("--normalized-target-risks", default=",".join(str(x) for x in DEFAULT_NORMALIZED_TARGET_RISKS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    equity = pnls.cumsum()
    equity = pd.concat([pd.Series([0.0]), equity], ignore_index=True)
    return float((equity - equity.cummax()).min())


def kelly_fraction(probability: pd.Series, payoff_ratio: float) -> pd.Series:
    q = 1.0 - probability
    raw = (payoff_ratio * probability - q) / payoff_ratio
    return raw.clip(lower=0.0)


def scaled_kelly_multiplier(raw_kelly: pd.Series, scale: float, min_mult: float, max_mult: float) -> pd.Series:
    return (1.0 + raw_kelly * scale).clip(lower=min_mult, upper=max_mult)


def fit_scale_for_target(
    train_raw_kelly: pd.Series,
    target_multiplier: float,
    min_mult: float,
    max_mult: float,
) -> dict[str, float | bool]:
    min_possible = float(scaled_kelly_multiplier(train_raw_kelly, 0.0, min_mult, max_mult).mean())
    max_possible = float(scaled_kelly_multiplier(train_raw_kelly, 1e12, min_mult, max_mult).mean())
    if target_multiplier <= min_possible:
        scale = 0.0
        achievable = target_multiplier >= min_possible
    elif target_multiplier >= max_possible:
        scale = 1e12
        achievable = target_multiplier <= max_possible
    else:
        lo, hi = 0.0, 1.0
        while float(scaled_kelly_multiplier(train_raw_kelly, hi, min_mult, max_mult).mean()) < target_multiplier:
            hi *= 2.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            avg = float(scaled_kelly_multiplier(train_raw_kelly, mid, min_mult, max_mult).mean())
            if avg < target_multiplier:
                lo = mid
            else:
                hi = mid
        scale = hi
        achievable = True
    actual_train_avg = float(scaled_kelly_multiplier(train_raw_kelly, scale, min_mult, max_mult).mean())
    return {
        "scale": float(scale),
        "target_multiplier": float(target_multiplier),
        "actual_train_avg_multiplier": actual_train_avg,
        "min_possible_train_avg_multiplier": min_possible,
        "max_possible_train_avg_multiplier": max_possible,
        "target_achievable": bool(achievable),
    }


def score_dataset(df: pd.DataFrame, model_bundle: dict[str, Any]) -> pd.DataFrame:
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    target = model_bundle["target"]
    prob_col = f"prob_{target}"
    scored = df.copy()
    scored[prob_col] = model.predict_proba(scored[feature_columns])[:, 1]
    return scored


def attach_execution_metadata(scored: pd.DataFrame, execution_events_path: Path) -> pd.DataFrame:
    if not execution_events_path.exists():
        raise SystemExit(f"Missing execution events for integer sizing: {execution_events_path}")

    exec_df = pd.read_parquet(execution_events_path)
    exec_df["signal_ts"] = pd.to_datetime(exec_df["signal_ts"], utc=True)
    exec_df["ny_date_key"] = exec_df["ny_date"].astype(str)
    exec_df = exec_df[
        exec_df["orb_minutes"].eq(15)
        & exec_df["exit_mode"].eq("tp_2r_or_time")
        & exec_df["side_mode"].isin(["long", "short"])
    ].copy()
    exec_df["join_side"] = exec_df["side"].map({"LONG": "UP", "SHORT": "DOWN"})
    exec_df = exec_df.drop_duplicates(["ny_date_key", "signal_ts", "join_side"], keep="first")

    base = scored.copy()
    base["signal_ts"] = pd.to_datetime(base["signal_ts"], utc=True)
    base["ny_date_key"] = base["ny_date"].astype(str)
    merged = base.merge(
        exec_df[
            [
                "ny_date_key",
                "signal_ts",
                "join_side",
                "entry_price",
                "stop_reference",
                "entry_risk_pts",
                "risk_per_contract_usd",
            ]
        ],
        left_on=["ny_date_key", "signal_ts", "side"],
        right_on=["ny_date_key", "signal_ts", "join_side"],
        how="left",
    )
    merged["execution_metadata_missing"] = merged["risk_per_contract_usd"].isna()
    missing = merged["risk_per_contract_usd"].isna()
    fallback = merged["pnl_per_contract_usd"] / merged["r_multiple"]
    fallback = fallback.where(np.isfinite(fallback) & fallback.gt(0))
    merged.loc[missing, "risk_per_contract_usd"] = fallback.loc[missing]
    if bool(missing.any()):
        still_missing = merged["risk_per_contract_usd"].isna()
        if bool(still_missing.any()):
            sample = merged.loc[still_missing, ["event_id", "ny_date", "side", "signal_ts"]].head(10).to_dict("records")
            raise SystemExit(f"Missing execution risk for {int(still_missing.sum())} rows. Sample: {sample}")
    return merged.drop(columns=["ny_date_key", "join_side"])


def add_integer_execution_columns(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["contracts_float"] = out["risk_usd"] / out["risk_per_contract_usd"]
    out["contracts_min1_floor"] = np.maximum(1, np.floor(out["contracts_float"])).astype(int)
    out["floor_risk_usd"] = out["contracts_min1_floor"] * out["risk_per_contract_usd"]
    out["floor_pnl_usd"] = out["r_multiple"] * out["floor_risk_usd"]
    out["contracts_minrisk_ceil"] = np.maximum(1, np.ceil(out["contracts_float"])).astype(int)
    out["integer_risk_usd"] = out["contracts_minrisk_ceil"] * out["risk_per_contract_usd"]
    out["integer_pnl_usd"] = out["r_multiple"] * out["integer_risk_usd"]
    out["desired_below_one_contract"] = out["contracts_float"] < 1.0
    out["integer_risk_over_desired"] = out["integer_risk_usd"] > (out["risk_usd"] + 1e-9)
    out["integer_risk_to_desired"] = out["integer_risk_usd"] / out["risk_usd"]
    return out


def build_variant_events(
    scored: pd.DataFrame,
    probability_col: str,
    base_risk: float,
    payoff_ratio: float,
    min_mult: float,
    max_mult: float,
    kelly_fractions: list[float],
    normalized_target_risks: list[float],
) -> pd.DataFrame:
    frames = []
    base = scored.copy()
    base["kelly_raw"] = kelly_fraction(base[probability_col], payoff_ratio)

    fixed = base.copy()
    fixed["variant"] = "fixed_1.00x"
    fixed["sizing_mode"] = "fixed"
    fixed["kelly_fraction"] = np.nan
    fixed["target_avg_risk_usd"] = np.nan
    fixed["normalization_scale"] = np.nan
    fixed["risk_multiplier"] = 1.0
    fixed["risk_usd"] = base_risk
    fixed["pnl_usd"] = fixed["r_multiple"] * fixed["risk_usd"]
    frames.append(fixed)

    for frac in kelly_fractions:
        cur = base.copy()
        cur["variant"] = f"basefloor_kelly_{frac:.2f}x"
        cur["sizing_mode"] = "base_floor_fractional_kelly"
        cur["kelly_fraction"] = frac
        cur["target_avg_risk_usd"] = np.nan
        cur["normalization_scale"] = np.nan
        cur["risk_multiplier"] = (1.0 + frac * cur["kelly_raw"]).clip(lower=min_mult, upper=max_mult)
        cur["risk_usd"] = base_risk * cur["risk_multiplier"]
        cur["pnl_usd"] = cur["r_multiple"] * cur["risk_usd"]
        frames.append(cur)

    train_raw = base.loc[base["split"] == "train", "kelly_raw"]
    if train_raw.empty:
        raise SystemExit("Cannot fit normalized Kelly scale: train split is empty.")
    for target_risk in normalized_target_risks:
        target_multiplier = float(target_risk / base_risk)
        scale_info = fit_scale_for_target(train_raw, target_multiplier, min_mult, max_mult)
        cur = base.copy()
        cur["variant"] = f"norm_target_{int(target_risk)}"
        cur["sizing_mode"] = "base_floor_normalized_kelly"
        cur["kelly_fraction"] = np.nan
        cur["target_avg_risk_usd"] = float(target_risk)
        cur["normalization_scale"] = float(scale_info["scale"])
        cur["risk_multiplier"] = scaled_kelly_multiplier(cur["kelly_raw"], float(scale_info["scale"]), min_mult, max_mult)
        cur["risk_usd"] = base_risk * cur["risk_multiplier"]
        cur["pnl_usd"] = cur["r_multiple"] * cur["risk_usd"]
        cur["target_achievable_on_train"] = bool(scale_info["target_achievable"])
        cur["max_possible_train_avg_multiplier"] = float(scale_info["max_possible_train_avg_multiplier"])
        frames.append(cur)

    keep = [
        "variant",
        "sizing_mode",
        "kelly_fraction",
        "target_avg_risk_usd",
        "normalization_scale",
        "event_id",
        "ny_date",
        "split",
        "side",
        "signal_ts",
        "entry_ts",
        "exit_ts",
        "exit_reason",
        probability_col,
        "kelly_raw",
        "entry_price",
        "stop_reference",
        "entry_risk_pts",
        "risk_per_contract_usd",
        "pnl_per_contract_usd",
        "execution_metadata_missing",
        "risk_multiplier",
        "risk_usd",
        "pnl_usd",
        "contracts_float",
        "contracts_min1_floor",
        "floor_risk_usd",
        "floor_pnl_usd",
        "contracts_minrisk_ceil",
        "integer_risk_usd",
        "integer_pnl_usd",
        "desired_below_one_contract",
        "integer_risk_over_desired",
        "integer_risk_to_desired",
        "target_achievable_on_train",
        "max_possible_train_avg_multiplier",
        "r_multiple",
        "success_2r",
        "positive_eod",
        "outcome_bucket",
    ]
    all_events = pd.concat(frames, ignore_index=True)
    for col in ["target_achievable_on_train", "max_possible_train_avg_multiplier"]:
        if col not in all_events.columns:
            all_events[col] = np.nan
    all_events = add_integer_execution_columns(all_events)
    return all_events[keep].sort_values(["variant", "signal_ts", "side"]).reset_index(drop=True)


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, split), group in events.groupby(["variant", "split"], sort=True):
        group = group.sort_values("signal_ts")
        pnl = group["pnl_usd"]
        integer_pnl = group["integer_pnl_usd"]
        integer_dd = max_drawdown(integer_pnl)
        rows.append(
            {
                "variant": variant,
                "sizing_mode": str(group["sizing_mode"].iloc[0]),
                "split": split,
                "rows": int(len(group)),
                "pnl_usd": float(pnl.sum()),
                "max_dd_usd": max_drawdown(pnl),
                "avg_pnl_usd": float(pnl.mean()),
                "return_dd": float(pnl.sum() / abs(max_drawdown(pnl))) if max_drawdown(pnl) < 0 else None,
                "integer_pnl_usd": float(integer_pnl.sum()),
                "integer_max_dd_usd": integer_dd,
                "integer_avg_pnl_usd": float(integer_pnl.mean()),
                "integer_return_dd": float(integer_pnl.sum() / abs(integer_dd)) if integer_dd < 0 else None,
                "success_2r_rate": float(group["success_2r"].mean()),
                "positive_eod_rate": float(group["positive_eod"].mean()),
                "avg_probability": float(group.filter(like="prob_").iloc[:, 0].mean()),
                "avg_kelly_raw": float(group["kelly_raw"].mean()),
                "avg_risk_multiplier": float(group["risk_multiplier"].mean()),
                "min_risk_multiplier": float(group["risk_multiplier"].min()),
                "max_risk_multiplier": float(group["risk_multiplier"].max()),
                "avg_risk_usd": float(group["risk_usd"].mean()),
                "avg_contracts_float": float(group["contracts_float"].mean()),
                "avg_contracts_min1_floor": float(group["contracts_min1_floor"].mean()),
                "avg_contracts_minrisk_ceil": float(group["contracts_minrisk_ceil"].mean()),
                "avg_integer_risk_usd": float(group["integer_risk_usd"].mean()),
                "avg_floor_risk_usd": float(group["floor_risk_usd"].mean()),
                "desired_below_one_contract_rate": float(group["desired_below_one_contract"].mean()),
                "integer_risk_over_desired_rate": float(group["integer_risk_over_desired"].mean()),
                "avg_integer_risk_to_desired": float(group["integer_risk_to_desired"].mean()),
                "execution_metadata_missing_rate": float(group["execution_metadata_missing"].mean()),
                "target_avg_risk_usd": None if pd.isna(group["target_avg_risk_usd"].iloc[0]) else float(group["target_avg_risk_usd"].iloc[0]),
                "normalization_scale": None if pd.isna(group["normalization_scale"].iloc[0]) else float(group["normalization_scale"].iloc[0]),
                "target_achievable_on_train": None if pd.isna(group["target_achievable_on_train"].iloc[0]) else bool(group["target_achievable_on_train"].iloc[0]),
                "max_possible_train_avg_multiplier": None
                if pd.isna(group["max_possible_train_avg_multiplier"].iloc[0])
                else float(group["max_possible_train_avg_multiplier"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "variant"]).reset_index(drop=True)


def split_manifest(summary: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for split, group in summary.groupby("split", sort=True):
        out[str(split)] = group.sort_values("variant").to_dict("records")
    return out


def write_report(path: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# MNQ ORB Base-Floor Kelly Overlay V2",
        "",
        f"Created: `{manifest['created_at']}`",
        "",
        "## Contract",
        "",
        f"- Base risk: `${manifest['base_risk_usd']:.0f}`",
        f"- Payoff ratio `b`: `{manifest['payoff_ratio']}`",
        f"- Break-even probability: `{manifest['break_even_probability']:.4f}`",
        f"- Min/max risk multiplier: `{manifest['min_risk_multiplier']}` / `{manifest['max_risk_multiplier']}`",
        f"- Kelly fractions: `{manifest['kelly_fractions']}`",
        f"- Normalized target risks: `{manifest['normalized_target_risks']}`",
        "",
        "Base-floor fractional Kelly formula:",
        "",
        "```text",
        manifest["formula"],
        "```",
        "",
        "Base-floor normalized Kelly formula:",
        "",
        "```text",
        manifest["normalized_formula"],
        "```",
        "",
        "Continuous risk sizing: `pnl_usd = r_multiple * risk_usd`.",
        "",
        "Executable integer sizing:",
        "",
        "```text",
        "contracts_minrisk_ceil = max(1, ceil(risk_usd / risk_per_contract_usd))",
        "integer_pnl_usd = r_multiple * contracts_minrisk_ceil * risk_per_contract_usd",
        "```",
        "",
    ]

    for split in ["train", "validation", "holdout"]:
        cur = summary[summary["split"] == split].copy()
        if cur.empty:
            continue
        lines.append(f"## {split.title()}")
        lines.append("")
        lines.append("| Variant | Rows | Cont PnL | Cont DD | Cont R/DD | Int PnL | Int DD | Int R/DD | Avg desired risk | Avg int risk | Under 1ct |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in cur.sort_values("variant").itertuples(index=False):
            ret_dd = "" if pd.isna(row.return_dd) else f"{row.return_dd:.2f}"
            int_ret_dd = "" if pd.isna(row.integer_return_dd) else f"{row.integer_return_dd:.2f}"
            lines.append(
                f"| `{row.variant}` | {row.rows} | ${row.pnl_usd:,.0f} | ${row.max_dd_usd:,.0f} | "
                f"{ret_dd} | ${row.integer_pnl_usd:,.0f} | ${row.integer_max_dd_usd:,.0f} | "
                f"{int_ret_dd} | ${row.avg_risk_usd:,.0f} | ${row.avg_integer_risk_usd:,.0f} | "
                f"{row.desired_below_one_contract_rate:.1%} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Readout",
            "",
            "- Desired risk is floored at the base $500 risk; Kelly only adds risk above the baseline.",
            "- Normalized variants fit their scaling on train only, then apply the same scale to validation/holdout.",
            "- Max multiplier caps desired risk; default cap is 2.0x base risk.",
            "- Integer MNQ execution is the practical constraint: executable contracts are rounded up so actual risk is at least the desired risk.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    dataset_path = project_path(cfg["outputs"]["breakout_quality"])
    events_path = project_path(cfg["outputs"]["kelly_overlay_events"])
    summary_path = project_path(cfg["outputs"]["kelly_overlay_summary"])
    manifest_path = project_path(cfg["outputs"]["kelly_overlay_manifest"])
    model_path = project_path(args.model_artifact)
    execution_events_path = project_path(args.execution_events)

    if not dataset_path.exists():
        raise SystemExit(f"Missing breakout-quality dataset: {dataset_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing model artifact: {model_path}")
    for path in [events_path, summary_path, manifest_path]:
        if path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing artifact: {path}")

    df = pd.read_parquet(dataset_path)
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    bundle = joblib.load(model_path)
    scored = attach_execution_metadata(score_dataset(df, bundle), execution_events_path)
    probability_col = f"prob_{bundle['target']}"
    kelly_fractions = parse_float_list(args.kelly_fractions)
    normalized_target_risks = parse_float_list(args.normalized_target_risks)
    events = build_variant_events(
        scored=scored,
        probability_col=probability_col,
        base_risk=args.base_risk_usd,
        payoff_ratio=args.payoff_ratio,
        min_mult=args.min_risk_multiplier,
        max_mult=args.max_risk_multiplier,
        kelly_fractions=kelly_fractions,
        normalized_target_risks=normalized_target_risks,
    )
    summary = summarize(events)

    events_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_path, index=False)
    summary.to_parquet(summary_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset": str(dataset_path),
        "model_artifact": str(model_path),
        "execution_events": str(execution_events_path),
        "model_target": bundle["target"],
        "probability_col": probability_col,
        "events_output": str(events_path),
        "summary_output": str(summary_path),
        "base_risk_usd": args.base_risk_usd,
        "payoff_ratio": args.payoff_ratio,
        "break_even_probability": 1.0 / (args.payoff_ratio + 1.0),
        "min_risk_multiplier": args.min_risk_multiplier,
        "max_risk_multiplier": args.max_risk_multiplier,
        "kelly_fractions": kelly_fractions,
        "normalized_target_risks": normalized_target_risks,
        "formula": "risk_multiplier = clip(1 + kelly_fraction * max(0, (b*p - (1-p))/b), min=1.0, max)",
        "normalized_formula": "scale is fit on train only; risk_multiplier = clip(1 + scale * max(0, (b*p - (1-p))/b), min=1.0, max)",
        "continuous_risk_note": "PnL is simulated as r_multiple * risk_usd.",
        "integer_execution_note": "Executable MNQ sizing uses max(1, ceil(risk_usd / risk_per_contract_usd)) so actual risk is at least desired risk.",
        "summary": split_manifest(summary),
    }
    report_path = project_path(cfg["outputs"]["model_dir"]) / "risk_adjusted_v1_kelly_overlay_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report_path, summary, manifest)
    manifest["report"] = str(report_path)
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "summary": str(summary_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
