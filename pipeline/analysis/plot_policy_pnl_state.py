"""
Plot current holdout PnL state for ORB_v1.0 policies.

Output:
  model/ORB_v1.0/POLICY_PNL_STATE.png
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DM_PATH = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"
REV_MODEL_PATH = ROOT / "model/ORB_v1.0/lgbm_rev_1r2_120m.txt"
CONT_MODEL_PATH = ROOT / "model/ORB_v1.0/lgbm_cont_1r2_120m.txt"
OUT_PNG = ROOT / "model/ORB_v1.0/POLICY_PNL_STATE.png"
OUT_CSV = ROOT / "model/ORB_v1.0/POLICY_PNL_STATE_METRICS.csv"
OUT_TOPSTEP_PNG = ROOT / "model/ORB_v1.0/POLICY_PNL_TOPSTEP.png"

TARGET = "y_1r2_120m"
HOLDOUT_FROM = "2024-01-01"
CALIB_FROM = "2022-01-01"
CALIB_TO = "2023-12-31"
RR = 2.0
COST_R = 0.07
TOPSTEP_50K_DD = 2000.0
TOPSTEP_100K_DD = 3000.0

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


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["session"] = out["session"].map({"tokyo": 0, "london": 1, "us": 2})
    out["orb_tf"] = out["orb_tf"].map({"5m": 5, "15m": 15, "30m": 30})
    return out


def side_frame(df: pd.DataFrame, side: str, model: lgb.Booster) -> pd.DataFrame:
    s = df[df["side"] == side].copy()
    s = encode(s).dropna(subset=FEATURES + [TARGET]).copy()
    s["prob"] = model.predict(s[FEATURES])
    s = s[EVENT_KEY + ["prob", TARGET, "adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"]]
    s = s.rename(columns={"prob": f"prob_{side}", TARGET: f"y_{side}"})
    return s


def net_r(y: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(y).astype(int)
    return np.where(arr == 1, RR, -1.0) - COST_R


def drawdown_r(equity_r: pd.Series) -> pd.Series:
    peak = equity_r.cummax()
    return peak - equity_r


def main() -> None:
    dm = pd.read_parquet(DM_PATH).copy()
    dm["date"] = pd.to_datetime(dm["date"])

    rev_model = lgb.Booster(model_file=str(REV_MODEL_PATH))
    cont_model = lgb.Booster(model_file=str(CONT_MODEL_PATH))

    rev = side_frame(dm, "rev", rev_model)
    cont = side_frame(dm, "cont", cont_model)

    events = rev.merge(
        cont,
        on=EVENT_KEY + ["adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"],
        how="inner",
    ).sort_values(["date", "breakout_ts"]).reset_index(drop=True)

    calib = events[(events["date"] >= CALIB_FROM) & (events["date"] <= CALIB_TO)].copy()
    holdout = events[events["date"] >= HOLDOUT_FROM].copy().reset_index(drop=True)

    t_rev = float(np.percentile(calib["prob_rev"], 60))
    t_cont = float(np.percentile(calib["prob_cont"], 60))
    q75_rev = float(np.percentile(calib["prob_rev"], 75))
    q75_cont = float(np.percentile(calib["prob_cont"], 75))

    trend_sign = np.sign(holdout["ema_slope_1h"].fillna(0))
    aligned_rev = trend_sign == -holdout["breakout_side"]
    aligned_cont = trend_sign == holdout["breakout_side"]

    rev_gate = (holdout["prob_rev"] >= t_rev) & aligned_rev & (
        (holdout["adx_14_15m"] >= 30) | (holdout["prob_rev"] >= q75_rev)
    )
    cont_gate = (holdout["prob_cont"] >= t_cont) & aligned_cont & (
        (holdout["adx_14_15m"] < 50) | (holdout["prob_cont"] >= q75_cont)
    )

    decision = np.full(len(holdout), "skip", dtype=object)
    decision[rev_gate & ~cont_gate] = "rev"
    decision[cont_gate & ~rev_gate] = "cont"
    both = np.where((rev_gate & cont_gate).values)[0]
    if len(both):
        pick_rev = holdout.iloc[both]["prob_rev"].values >= holdout.iloc[both]["prob_cont"].values
        decision[both[pick_rev]] = "rev"
        decision[both[~pick_rev]] = "cont"
    holdout["decision"] = decision

    # Event-level equity (always policies can trade every event)
    holdout["r_always_rev"] = net_r(holdout["y_rev"])
    holdout["r_always_cont"] = net_r(holdout["y_cont"])
    holdout["r_max_prob"] = np.where(holdout["prob_rev"] >= holdout["prob_cont"], holdout["r_always_rev"], holdout["r_always_cont"])
    holdout["r_dynamic"] = np.where(
        holdout["decision"] == "rev",
        holdout["r_always_rev"],
        np.where(holdout["decision"] == "cont", holdout["r_always_cont"], 0.0),
    )

    holdout["eq_always_rev"] = holdout["r_always_rev"].cumsum()
    holdout["eq_always_cont"] = holdout["r_always_cont"].cumsum()
    holdout["eq_max_prob"] = holdout["r_max_prob"].cumsum()
    holdout["eq_dynamic"] = holdout["r_dynamic"].cumsum()
    holdout["dd_dynamic"] = drawdown_r(holdout["eq_dynamic"])

    # Metrics snapshot
    metrics = pd.DataFrame(
        [
            ("always_rev", float(holdout["eq_always_rev"].iloc[-1]), float(drawdown_r(holdout["eq_always_rev"]).max()), int((holdout["r_always_rev"] < 0).sum())),
            ("always_cont", float(holdout["eq_always_cont"].iloc[-1]), float(drawdown_r(holdout["eq_always_cont"]).max()), int((holdout["r_always_cont"] < 0).sum())),
            ("max_prob", float(holdout["eq_max_prob"].iloc[-1]), float(drawdown_r(holdout["eq_max_prob"]).max()), int((holdout["r_max_prob"] < 0).sum())),
            ("dynamic_policy", float(holdout["eq_dynamic"].iloc[-1]), float(holdout["dd_dynamic"].max()), int((holdout["r_dynamic"] < 0).sum())),
        ],
        columns=["policy", "cum_net_r", "max_dd_r", "loss_events"],
    )
    metrics.to_csv(OUT_CSV, index=False)

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)

    x = np.arange(len(holdout))
    axes[0].plot(x, holdout["eq_always_rev"], label="Always Reversal", linewidth=1.6, color="#cc5500")
    axes[0].plot(x, holdout["eq_always_cont"], label="Always Continuation", linewidth=1.6, color="#3f7d20")
    axes[0].plot(x, holdout["eq_max_prob"], label="Max Prob (No Gate)", linewidth=1.6, color="#1f4e79")
    axes[0].plot(x, holdout["eq_dynamic"], label="Dynamic Rev/Cont/Skip", linewidth=2.4, color="#0a9396")
    axes[0].axhline(0, color="#444444", linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("Cumulative Net R")
    axes[0].set_title("ORB_v1.0 Holdout State (2024+): Equity Curves by Policy")
    axes[0].legend(loc="upper left", ncol=2, frameon=True)

    axes[1].plot(x, holdout["dd_dynamic"], color="#9b2226", linewidth=2.0, label="Dynamic Policy Drawdown (R)")
    axes[1].set_ylabel("Drawdown (R)")
    axes[1].set_xlabel("Holdout Events (chronological)")
    axes[1].axhline(93.19, color="#bb3e03", linestyle="--", linewidth=1.2, alpha=0.7, label="Observed Max DD ≈ 93.19R")
    axes[1].legend(loc="upper left", frameon=True)

    text = (
        f"Trades(dynamic): {(holdout['decision']!='skip').sum()} / {len(holdout)} ({(holdout['decision']!='skip').mean():.1%})\\n"
        f"Final Net R(dynamic): {holdout['eq_dynamic'].iloc[-1]:.2f}R | Max DD(dynamic): {holdout['dd_dynamic'].max():.2f}R\\n"
        f"Thresholds: rev p60={t_rev:.3f}, cont p60={t_cont:.3f}"
    )
    axes[0].text(
        0.99,
        0.02,
        text,
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#aaaaaa"),
    )

    fig.savefig(OUT_PNG, dpi=160)

    # Topstep-style USD view based on observed dynamic max DD in R.
    max_dd_r = float(holdout["dd_dynamic"].max())
    risk_50 = TOPSTEP_50K_DD / max_dd_r if max_dd_r > 0 else np.nan
    risk_100 = TOPSTEP_100K_DD / max_dd_r if max_dd_r > 0 else np.nan

    eq_usd_50 = holdout["eq_dynamic"] * risk_50
    dd_usd_50 = holdout["dd_dynamic"] * risk_50
    eq_usd_100 = holdout["eq_dynamic"] * risk_100
    dd_usd_100 = holdout["dd_dynamic"] * risk_100

    fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes2[0].plot(x, eq_usd_50, color="#0a9396", linewidth=2.2, label=f"Dynamic Equity (50K sizing, 1R=${risk_50:.2f})")
    axes2[0].plot(x, eq_usd_100, color="#005f73", linewidth=2.2, label=f"Dynamic Equity (100K sizing, 1R=${risk_100:.2f})")
    axes2[0].axhline(0, color="#444444", linewidth=1.0, alpha=0.7)
    axes2[0].set_ylabel("Cumulative Net PnL (USD)")
    axes2[0].set_title("Dynamic Policy PnL in USD Under Topstep DD Sizing")
    axes2[0].legend(loc="upper left", frameon=True)

    axes2[1].plot(x, dd_usd_50, color="#ee9b00", linewidth=2.0, label="Drawdown USD (50K sizing)")
    axes2[1].plot(x, dd_usd_100, color="#bb3e03", linewidth=2.0, label="Drawdown USD (100K sizing)")
    axes2[1].axhline(TOPSTEP_50K_DD, color="#ee9b00", linestyle="--", linewidth=1.2, alpha=0.8, label="50K DD limit $2,000")
    axes2[1].axhline(TOPSTEP_100K_DD, color="#bb3e03", linestyle="--", linewidth=1.2, alpha=0.8, label="100K DD limit $3,000")
    axes2[1].set_ylabel("Drawdown (USD)")
    axes2[1].set_xlabel("Holdout Events (chronological)")
    axes2[1].legend(loc="upper left", frameon=True, ncol=2)

    fig2.savefig(OUT_TOPSTEP_PNG, dpi=160)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_TOPSTEP_PNG}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
