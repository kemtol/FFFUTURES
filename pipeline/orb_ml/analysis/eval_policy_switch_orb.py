"""
Evaluate dynamic ORB policy: choose rev/cont/skip using two side models.

Inputs:
  - model/ORB_v1.0/lgbm_rev_1r2_120m.txt
  - model/ORB_v1.0/lgbm_cont_1r2_120m.txt
  - data/Level_2_Datamart/training_datamart_orb.parquet

Output:
  - model/ORB_v1.0/POLICY_SWITCH_REPORT.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DM_PATH = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"
MODEL_DIR = ROOT / "model/ORB_v1.0"
REV_MODEL_PATH = MODEL_DIR / "lgbm_rev_1r2_120m.txt"
CONT_MODEL_PATH = MODEL_DIR / "lgbm_cont_1r2_120m.txt"
OUT_REPORT = MODEL_DIR / "POLICY_SWITCH_REPORT.md"

TARGET = "y_1r2_120m"
HOLDOUT_FROM = "2024-01-01"
CALIB_FROM = "2022-01-01"
CALIB_TO = "2023-12-31"
COST_R = 0.07
RR = 2.0

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

EVENT_KEY = ["date", "breakout_ts", "session", "orb_tf", "breakout_side"]


@dataclass
class TradeStats:
    n: int
    win_rate: float
    exp_net: float
    exp_gross: float
    pf_net: float
    sharpe_trade_net: float
    max_loss_streak: int


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["session"] = out["session"].map({"tokyo": 0, "london": 1, "us": 2})
    out["orb_tf"] = out["orb_tf"].map({"5m": 5, "15m": 15, "30m": 30})
    return out


def max_consecutive_losses(net_r: np.ndarray) -> int:
    max_streak = 0
    streak = 0
    for r in net_r:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def calc_stats_from_binary(y: np.ndarray) -> TradeStats:
    if y.size == 0:
        return TradeStats(0, math.nan, math.nan, math.nan, math.nan, math.nan, 0)

    gross = np.where(y == 1, RR, -1.0)
    net = gross - COST_R

    wr = float(y.mean())
    exp_gross = float(gross.mean())
    exp_net = float(net.mean())

    pos = float(net[net > 0].sum())
    neg = float(-net[net < 0].sum())
    pf_net = pos / neg if neg > 0 else math.nan

    if len(net) > 1 and np.std(net, ddof=1) > 0:
        sharpe_trade_net = float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(len(net)))
    else:
        sharpe_trade_net = math.nan

    return TradeStats(
        n=int(len(y)),
        win_rate=wr,
        exp_net=exp_net,
        exp_gross=exp_gross,
        pf_net=pf_net,
        sharpe_trade_net=sharpe_trade_net,
        max_loss_streak=max_consecutive_losses(net),
    )


def f(v: float, nd: int = 3) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:.{nd}f}"


def side_frame(df: pd.DataFrame, side: str, model: lgb.Booster) -> pd.DataFrame:
    s = df[df["side"] == side].copy()
    s = encode(s).dropna(subset=FEATURES + [TARGET]).copy()
    s["prob"] = model.predict(s[FEATURES])
    s = s[EVENT_KEY + ["prob", TARGET, "adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"]]
    s = s.rename(columns={"prob": f"prob_{side}", TARGET: f"y_{side}"})
    return s


def evaluate_policy(events: pd.DataFrame, t_rev: float, t_cont: float, q75_rev: float, q75_cont: float) -> pd.DataFrame:
    e = events.copy()
    trend_sign = np.sign(e["ema_slope_1h"].fillna(0))
    e["aligned_rev"] = trend_sign == -e["breakout_side"]
    e["aligned_cont"] = trend_sign == e["breakout_side"]

    rev_gate = (e["prob_rev"] >= t_rev) & e["aligned_rev"] & ((e["adx_14_15m"] >= 30) | (e["prob_rev"] >= q75_rev))
    cont_gate = (e["prob_cont"] >= t_cont) & e["aligned_cont"] & ((e["adx_14_15m"] < 50) | (e["prob_cont"] >= q75_cont))

    choice = np.full(len(e), "skip", dtype=object)
    rev_only = rev_gate & ~cont_gate
    cont_only = cont_gate & ~rev_gate
    both = rev_gate & cont_gate

    choice[rev_only.values] = "rev"
    choice[cont_only.values] = "cont"
    both_idx = np.where(both.values)[0]
    if len(both_idx):
        pick_rev = e.iloc[both_idx]["prob_rev"].values >= e.iloc[both_idx]["prob_cont"].values
        choice[both_idx[pick_rev]] = "rev"
        choice[both_idx[~pick_rev]] = "cont"

    e["decision"] = choice
    e["y_decision"] = np.where(e["decision"] == "rev", e["y_rev"], np.where(e["decision"] == "cont", e["y_cont"], np.nan))
    return e


def row(name: str, s: TradeStats) -> dict:
    return {
        "policy": name,
        "n": s.n,
        "win_rate": s.win_rate,
        "exp_net_R": s.exp_net,
        "exp_gross_R": s.exp_gross,
        "pf_net": s.pf_net,
        "sharpe_trade_net": s.sharpe_trade_net,
        "max_loss_streak": s.max_loss_streak,
    }


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No data_"
    header = "| " + " | ".join(cols) + " |\n"
    sep = "|" + "|".join(["---"] * len(cols)) + "|\n"
    lines = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return header + sep + "\n".join(lines)


def main() -> None:
    dm = pd.read_parquet(DM_PATH).copy()
    dm["date"] = pd.to_datetime(dm["date"])

    rev_model = lgb.Booster(model_file=str(REV_MODEL_PATH))
    cont_model = lgb.Booster(model_file=str(CONT_MODEL_PATH))

    rev_all = side_frame(dm, "rev", rev_model)
    cont_all = side_frame(dm, "cont", cont_model)

    ev = rev_all.merge(cont_all, on=EVENT_KEY + ["adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"], how="inner")
    ev = ev.sort_values(["date", "breakout_ts"]).reset_index(drop=True)
    ev["year"] = ev["date"].dt.year.astype(str)
    ev["vwap_bucket"] = np.where(ev["price_vs_vwap_pct"] >= 0, "above_vwap", "below_vwap")

    calib = ev[(ev["date"] >= CALIB_FROM) & (ev["date"] <= CALIB_TO)].copy()
    holdout = ev[ev["date"] >= HOLDOUT_FROM].copy()

    t_rev = float(np.percentile(calib["prob_rev"], 60))
    t_cont = float(np.percentile(calib["prob_cont"], 60))
    q75_rev = float(np.percentile(calib["prob_rev"], 75))
    q75_cont = float(np.percentile(calib["prob_cont"], 75))

    holdout_eval = evaluate_policy(holdout, t_rev, t_cont, q75_rev, q75_cont)
    traded = holdout_eval[holdout_eval["decision"] != "skip"].copy()

    y_rev = holdout_eval["y_rev"].astype(int).to_numpy()
    y_cont = holdout_eval["y_cont"].astype(int).to_numpy()
    y_dyn = traded["y_decision"].astype(int).to_numpy()
    y_maxprob = np.where(holdout_eval["prob_rev"] >= holdout_eval["prob_cont"], holdout_eval["y_rev"], holdout_eval["y_cont"]).astype(int)

    overall_rows = pd.DataFrame(
        [
            row("always_rev", calc_stats_from_binary(y_rev)),
            row("always_cont", calc_stats_from_binary(y_cont)),
            row("max_prob_no_gate", calc_stats_from_binary(y_maxprob)),
            row("dynamic_rev_cont_skip", calc_stats_from_binary(y_dyn)),
        ]
    )

    yearly_rows = []
    for yr, g in holdout_eval.groupby("year"):
        gt = g[g["decision"] != "skip"]
        if len(gt) == 0:
            continue
        yearly_rows.append(row(str(yr), calc_stats_from_binary(gt["y_decision"].astype(int).to_numpy())))
    yearly_df = pd.DataFrame(yearly_rows)

    decision_mix = holdout_eval["decision"].value_counts(normalize=True).rename_axis("decision").reset_index(name="share")

    # Regime slices for traded decisions
    traded["trend_bucket"] = np.where(
        np.sign(traded["ema_slope_1h"].fillna(0)) == traded["breakout_side"],
        "trend_breakout",
        "trend_reversal",
    )
    traded["adx_bucket"] = pd.cut(
        traded["adx_14_15m"],
        bins=[-np.inf, 20, 30, 50, np.inf],
        labels=["<20", "20-30", "30-50", ">50"],
        right=True,
    ).astype(str)

    regime_rows = []
    for name, g in traded.groupby("adx_bucket"):
        regime_rows.append(row(f"adx_{name}", calc_stats_from_binary(g["y_decision"].astype(int).to_numpy())))
    for name, g in traded.groupby("trend_bucket"):
        regime_rows.append(row(name, calc_stats_from_binary(g["y_decision"].astype(int).to_numpy())))
    for name, g in traded.groupby("vwap_bucket"):
        regime_rows.append(row(name, calc_stats_from_binary(g["y_decision"].astype(int).to_numpy())))
    regime_df = pd.DataFrame(regime_rows)

    dyn_stats = overall_rows[overall_rows["policy"] == "dynamic_rev_cont_skip"].iloc[0]
    north_pf = dyn_stats["pf_net"] > 1.3 if not pd.isna(dyn_stats["pf_net"]) else False
    north_sharpe = dyn_stats["sharpe_trade_net"] > 1.0 if not pd.isna(dyn_stats["sharpe_trade_net"]) else False
    north_streak = dyn_stats["max_loss_streak"] < 8

    report = f"""# Policy Switch Evaluation — ORB_v1.0

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Policy

- Models: `lgbm_rev_1r2_120m` + `lgbm_cont_1r2_120m`
- Holdout: `{HOLDOUT_FROM}` onward
- Calibration window: `{CALIB_FROM}` to `{CALIB_TO}`
- Thresholds:
  - `t_rev` (p60): `{t_rev:.6f}`
  - `t_cont` (p60): `{t_cont:.6f}`
  - `q75_rev`: `{q75_rev:.6f}`
  - `q75_cont`: `{q75_cont:.6f}`
- Decision logic:
  - `rev` candidate if `prob_rev >= t_rev`, trend aligns with reversal, and (`ADX>=30` or `prob_rev>=q75_rev`)
  - `cont` candidate if `prob_cont >= t_cont`, trend aligns with breakout, and (`ADX<50` or `prob_cont>=q75_cont`)
  - if both candidates valid: pick higher probability
  - if none valid: `skip`

## Overall (Holdout 2024+)

{md_table(overall_rows, ["policy", "n", "win_rate", "exp_net_R", "pf_net", "sharpe_trade_net", "max_loss_streak"])}

## Decision Mix (Dynamic Policy)

{md_table(decision_mix, ["decision", "share"])}

## Dynamic Policy by Year

{md_table(yearly_df, ["policy", "n", "win_rate", "exp_net_R", "pf_net", "max_loss_streak"])}

## Dynamic Policy by Regime

{md_table(regime_df, ["policy", "n", "win_rate", "exp_net_R", "pf_net", "max_loss_streak"])}

## North Star Check (Dynamic Policy)

- PF net > 1.3: `{north_pf}`
- Sharpe trade net > 1.0: `{north_sharpe}`
- Max loss streak < 8: `{north_streak}`
"""

    OUT_REPORT.write_text(report)
    print(f"Saved: {OUT_REPORT}")
    print(f"Holdout events: {len(holdout_eval)} | traded: {len(traded)} | skip: {(holdout_eval['decision']=='skip').sum()}")


if __name__ == "__main__":
    main()
