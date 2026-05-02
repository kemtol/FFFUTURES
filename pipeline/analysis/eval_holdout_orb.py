"""
Evaluate ORB reversal model on locked holdout (2024+).

Outputs:
  - model/ORB_vX.Y/HOLDOUT_REPORT.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DM_PATH = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"
MODEL_DIR = ROOT / "model/ORB_v1.0"
MODEL_PATH = MODEL_DIR / "lgbm_rev_1r2_120m.txt"
OUT_REPORT = MODEL_DIR / "HOLDOUT_REPORT.md"

TARGET = "y_1r2_120m"
HOLDOUT_FROM = "2024-01-01"
CALIB_FROM = "2022-01-01"
CALIB_TO = "2023-12-31"
RR = 2.0
COST_R = 0.07

FEATURES = [
    "orb_range_atr_ratio",
    "breakout_strength",
    "atr14_at_entry",
    "price_vs_vwap_pct",
    "adx_14_15m",
    "ema_slope_1h",
    "day_of_week",
    "time_in_session_min",
    "orb_tf",
    "session",
    "breakout_side",
]


@dataclass
class Stats:
    n: int
    win_rate: float
    exp_gross: float
    exp_net: float
    pf_gross: float
    pf_net: float
    sharpe_trade_net: float
    max_loss_streak: int


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["session"] = out["session"].map({"tokyo": 0, "london": 1, "us": 2})
    out["orb_tf"] = out["orb_tf"].map({"5m": 5, "15m": 15, "30m": 30})
    return out


def adx_bucket(x: float) -> str:
    if pd.isna(x):
        return "na"
    if x < 20:
        return "<20"
    if x < 30:
        return "20-30"
    if x <= 50:
        return "30-50"
    return ">50"


def max_consecutive_losses(net_r: np.ndarray) -> int:
    max_streak = 0
    streak = 0
    for r in net_r:
        if r < 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return max_streak


def calc_stats(df: pd.DataFrame, target: str = TARGET) -> Stats:
    y = df[target].dropna().astype(int).to_numpy()
    if y.size == 0:
        return Stats(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0)

    gross_r = np.where(y == 1, RR, -1.0)
    net_r = gross_r - COST_R
    win_rate = float(y.mean())
    exp_gross = float(gross_r.mean())
    exp_net = float(net_r.mean())

    gross_pos = float(gross_r[gross_r > 0].sum())
    gross_neg = float(-gross_r[gross_r < 0].sum())
    pf_gross = gross_pos / gross_neg if gross_neg > 0 else math.nan

    net_pos = float(net_r[net_r > 0].sum())
    net_neg = float(-net_r[net_r < 0].sum())
    pf_net = net_pos / net_neg if net_neg > 0 else math.nan

    if len(net_r) > 1 and np.std(net_r, ddof=1) > 0:
        sharpe_trade_net = float(np.mean(net_r) / np.std(net_r, ddof=1) * np.sqrt(len(net_r)))
    else:
        sharpe_trade_net = math.nan

    return Stats(
        n=int(len(y)),
        win_rate=win_rate,
        exp_gross=exp_gross,
        exp_net=exp_net,
        pf_gross=pf_gross,
        pf_net=pf_net,
        sharpe_trade_net=sharpe_trade_net,
        max_loss_streak=max_consecutive_losses(net_r),
    )


def to_row(label: str, s: Stats) -> dict:
    return {
        "bucket": label,
        "n": s.n,
        "win_rate": s.win_rate,
        "exp_gross_R": s.exp_gross,
        "exp_net_R": s.exp_net,
        "pf_gross": s.pf_gross,
        "pf_net": s.pf_net,
        "sharpe_trade_net": s.sharpe_trade_net,
        "max_loss_streak": s.max_loss_streak,
    }


def grouped_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        rows.append(to_row(str(key), calc_stats(g)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("n", ascending=False).reset_index(drop=True)
    return out


def format_float(v: float, nd: int = 3) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:.{nd}f}"


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No data_"
    head = "| " + " | ".join(cols) + " |\n"
    sep = "|" + "|".join(["---"] * len(cols)) + "|\n"
    lines = []
    for _, r in df.iterrows():
        row = []
        for c in cols:
            val = r[c]
            if isinstance(val, (float, np.floating)):
                row.append(format_float(float(val)))
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")
    return head + sep + "\n".join(lines)


def build_report(
    threshold: float,
    holdout_all_rev: Stats,
    holdout_sel_rev: Stats,
    holdout_all_cont: Stats,
    yearly_all: pd.DataFrame,
    yearly_sel: pd.DataFrame,
    adx_sel: pd.DataFrame,
    vwap_sel: pd.DataFrame,
    trend_sel: pd.DataFrame,
    side_yearly: pd.DataFrame,
    sel_share: float,
) -> str:
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    north_star_pf = holdout_sel_rev.pf_net > 1.3 if not pd.isna(holdout_sel_rev.pf_net) else False
    north_star_sharpe = holdout_sel_rev.sharpe_trade_net > 1.0 if not pd.isna(holdout_sel_rev.sharpe_trade_net) else False
    north_star_streak = holdout_sel_rev.max_loss_streak < 8

    return f"""# Holdout Evaluation — ORB_v1.0

Generated: {now}

## Setup

- Target: `{TARGET}` (reversal only model)
- Holdout window: `{HOLDOUT_FROM}` onward
- Threshold calibration window: `{CALIB_FROM}` to `{CALIB_TO}`
- Selection rule: `prob >= {threshold:.6f}` (top 40% equivalent on calibration set)
- Selection share on holdout: `{sel_share:.2%}`
- Transaction cost assumed: `{COST_R:.2f}R` per trade

## Headline (Holdout 2024+)

### Reversal (all signals, no model filter)

{md_table(pd.DataFrame([to_row("rev_all", holdout_all_rev)]), ["bucket", "n", "win_rate", "exp_gross_R", "exp_net_R", "pf_net", "sharpe_trade_net", "max_loss_streak"])}

### Reversal (model-filtered)

{md_table(pd.DataFrame([to_row("rev_model_top40", holdout_sel_rev)]), ["bucket", "n", "win_rate", "exp_gross_R", "exp_net_R", "pf_net", "sharpe_trade_net", "max_loss_streak"])}

### Continuation baseline (all signals)

{md_table(pd.DataFrame([to_row("cont_all", holdout_all_cont)]), ["bucket", "n", "win_rate", "exp_gross_R", "exp_net_R", "pf_net", "sharpe_trade_net", "max_loss_streak"])}

## North Star Check (on model-filtered holdout)

- PF net > 1.3: `{north_star_pf}`
- Sharpe trade net > 1.0: `{north_star_sharpe}`
- Max loss streak < 8: `{north_star_streak}`

## By Year — Reversal

### All reversal signals

{md_table(yearly_all, ["bucket", "n", "win_rate", "exp_net_R", "pf_net", "max_loss_streak"])}

### Model-filtered reversal

{md_table(yearly_sel, ["bucket", "n", "win_rate", "exp_net_R", "pf_net", "max_loss_streak"])}

## Regime Breakdown (Model-filtered Reversal)

### ADX bucket

{md_table(adx_sel, ["bucket", "n", "win_rate", "exp_net_R", "pf_net", "max_loss_streak"])}

### Price vs VWAP

{md_table(vwap_sel, ["bucket", "n", "win_rate", "exp_net_R", "pf_net", "max_loss_streak"])}

### 1H Trend Alignment

Bucket definition:
- `aligned_with_reversal`: `ema_slope_1h` searah posisi reversal
- `aligned_with_breakout`: `ema_slope_1h` searah breakout (lawan reversal)
- `flat_or_unknown`: slope 0/NA

{md_table(trend_sel, ["bucket", "n", "win_rate", "exp_net_R", "pf_net", "max_loss_streak"])}

## Reversal vs Continuation by Year (All signals)

{md_table(side_yearly, ["bucket", "n", "win_rate", "exp_net_R", "pf_net"])}
"""


def main() -> None:
    dm = pd.read_parquet(DM_PATH).copy()
    dm["date"] = pd.to_datetime(dm["date"])
    dm = dm.dropna(subset=[TARGET])
    dm_enc = encode(dm)

    model = lgb.Booster(model_file=str(MODEL_PATH))

    rev = dm_enc[dm_enc["side"] == "rev"].copy()
    rev = rev.dropna(subset=FEATURES + [TARGET]).copy()

    calib_mask = (rev["date"] >= CALIB_FROM) & (rev["date"] <= CALIB_TO)
    holdout_mask = rev["date"] >= HOLDOUT_FROM

    calib = rev[calib_mask].copy()
    holdout = rev[holdout_mask].copy()

    calib_prob = model.predict(calib[FEATURES])
    threshold = float(np.percentile(calib_prob, 60))

    holdout["prob"] = model.predict(holdout[FEATURES])
    holdout["selected"] = holdout["prob"] >= threshold
    holdout["year"] = holdout["date"].dt.year.astype(str)
    holdout["adx_bucket"] = holdout["adx_14_15m"].apply(adx_bucket)
    holdout["vwap_bucket"] = np.where(holdout["price_vs_vwap_pct"] >= 0, "above_vwap", "below_vwap")

    reversal_dir = -holdout["breakout_side"]
    holdout["trend_bucket"] = np.where(
        holdout["ema_slope_1h"].isna() | (holdout["ema_slope_1h"] == 0),
        "flat_or_unknown",
        np.where(
            np.sign(holdout["ema_slope_1h"]) == reversal_dir,
            "aligned_with_reversal",
            "aligned_with_breakout",
        ),
    )

    holdout_sel = holdout[holdout["selected"]].copy()

    cont_holdout = dm[(dm["side"] == "cont") & (dm["date"] >= HOLDOUT_FROM)].copy()

    holdout_all_rev_stats = calc_stats(holdout)
    holdout_sel_rev_stats = calc_stats(holdout_sel)
    holdout_all_cont_stats = calc_stats(cont_holdout)

    yearly_all = grouped_stats(holdout, "year")
    yearly_sel = grouped_stats(holdout_sel, "year")
    adx_sel = grouped_stats(holdout_sel, "adx_bucket")
    vwap_sel = grouped_stats(holdout_sel, "vwap_bucket")
    trend_sel = grouped_stats(holdout_sel, "trend_bucket")

    side_holdout = dm[dm["date"] >= HOLDOUT_FROM].copy()
    side_holdout["bucket"] = side_holdout["date"].dt.year.astype(str) + "_" + side_holdout["side"]
    side_yearly = grouped_stats(side_holdout, "bucket")

    sel_share = float(len(holdout_sel) / len(holdout)) if len(holdout) else 0.0
    report = build_report(
        threshold=threshold,
        holdout_all_rev=holdout_all_rev_stats,
        holdout_sel_rev=holdout_sel_rev_stats,
        holdout_all_cont=holdout_all_cont_stats,
        yearly_all=yearly_all,
        yearly_sel=yearly_sel,
        adx_sel=adx_sel,
        vwap_sel=vwap_sel,
        trend_sel=trend_sel,
        side_yearly=side_yearly,
        sel_share=sel_share,
    )
    OUT_REPORT.write_text(report)

    print(f"Saved: {OUT_REPORT}")
    print(f"Threshold (top40 calibration): {threshold:.6f}")
    print(f"Holdout reversal rows: {len(holdout)}")
    print(f"Selected reversal rows: {len(holdout_sel)} ({sel_share:.2%})")


if __name__ == "__main__":
    main()
