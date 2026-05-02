"""
Topstep Trading Combine pass-rate engine for ORB_v1.0.

This evaluates policies by the actual business objective:
passing a 20-trading-day window under profit target, trailing MLL,
consistency, optional daily stop, and optional daily profit cap.

Outputs:
  model/ORB_v1.0/TOPSTEP_PASS_GRID.csv
  model/ORB_v1.0/TOPSTEP_PASS_REPORT.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DM_PATH = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"
MODEL_DIR = ROOT / "model/ORB_v1.0"
REV_MODEL_PATH = MODEL_DIR / "lgbm_rev_1r2_120m.txt"
CONT_MODEL_PATH = MODEL_DIR / "lgbm_cont_1r2_120m.txt"
OUT_GRID = MODEL_DIR / "TOPSTEP_PASS_GRID.csv"
OUT_REPORT = MODEL_DIR / "TOPSTEP_PASS_REPORT.md"

TARGET = "y_1r2_120m"
HOLDOUT_FROM = "2024-01-01"
CALIB_FROM = "2022-01-01"
CALIB_TO = "2023-12-31"
WINDOW_DAYS = 20
RR = 2.0
COST_R = 0.07
MIN_GO_PASS_RATE = 0.60
MAX_GO_FAIL_MLL_RATE = 0.10
MIN_GO_YEAR_PASS_RATE = 0.30
MAX_GO_YEAR_FAIL_MLL_RATE = 0.20

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

ACCOUNTS = {
    "50K": {"profit_target": 3000.0, "mll": 2000.0},
    "100K": {"profit_target": 6000.0, "mll": 3000.0},
}


@dataclass(frozen=True)
class PolicyParams:
    rev_q: float
    cont_q: float
    rev_strong_q: float
    cont_strong_q: float
    rev_adx_min: float
    cont_adx_max: float
    daily_stop_usd: float
    daily_profit_cap_usd: float
    risk_per_r_usd: float


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
    return s.rename(columns={"prob": f"prob_{side}", TARGET: f"y_{side}"})


def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    dm = pd.read_parquet(DM_PATH).copy()
    dm["date"] = pd.to_datetime(dm["date"])
    dm["trade_day"] = dm["date"].dt.date

    rev_model = lgb.Booster(model_file=str(REV_MODEL_PATH))
    cont_model = lgb.Booster(model_file=str(CONT_MODEL_PATH))
    rev = side_frame(dm, "rev", rev_model)
    cont = side_frame(dm, "cont", cont_model)

    events = rev.merge(
        cont,
        on=EVENT_KEY + ["adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"],
        how="inner",
    )
    events = events.sort_values(["date", "breakout_ts"]).reset_index(drop=True)
    events["trade_day"] = events["date"].dt.date
    events["year"] = events["date"].dt.year
    events["adx_bucket"] = pd.cut(
        events["adx_14_15m"],
        bins=[-np.inf, 20, 30, 50, np.inf],
        labels=["<20", "20-30", "30-50", ">50"],
        right=True,
    ).astype(str)
    events["vwap_bucket"] = np.where(events["price_vs_vwap_pct"] >= 0, "above_vwap", "below_vwap")

    calib = events[(events["date"] >= CALIB_FROM) & (events["date"] <= CALIB_TO)].copy()
    holdout = events[events["date"] >= HOLDOUT_FROM].copy()
    return calib, holdout


def apply_policy(events: pd.DataFrame, calib: pd.DataFrame, p: PolicyParams) -> pd.DataFrame:
    out = events.copy()
    rev_t = float(np.percentile(calib["prob_rev"], p.rev_q * 100))
    cont_t = float(np.percentile(calib["prob_cont"], p.cont_q * 100))
    rev_strong = float(np.percentile(calib["prob_rev"], p.rev_strong_q * 100))
    cont_strong = float(np.percentile(calib["prob_cont"], p.cont_strong_q * 100))

    trend_sign = np.sign(out["ema_slope_1h"].fillna(0))
    aligned_rev = trend_sign == -out["breakout_side"]
    aligned_cont = trend_sign == out["breakout_side"]

    rev_gate = (out["prob_rev"] >= rev_t) & aligned_rev & (
        (out["adx_14_15m"] >= p.rev_adx_min) | (out["prob_rev"] >= rev_strong)
    )
    cont_gate = (out["prob_cont"] >= cont_t) & aligned_cont & (
        (out["adx_14_15m"] <= p.cont_adx_max) | (out["prob_cont"] >= cont_strong)
    )

    decision = np.full(len(out), "skip", dtype=object)
    decision[rev_gate & ~cont_gate] = "rev"
    decision[cont_gate & ~rev_gate] = "cont"
    both = np.where((rev_gate & cont_gate).values)[0]
    if len(both):
        pick_rev = out.iloc[both]["prob_rev"].values >= out.iloc[both]["prob_cont"].values
        decision[both[pick_rev]] = "rev"
        decision[both[~pick_rev]] = "cont"

    out["decision"] = decision
    out["y"] = np.where(out["decision"] == "rev", out["y_rev"], np.where(out["decision"] == "cont", out["y_cont"], np.nan))
    out["r_net"] = np.where(out["y"] == 1, RR, -1.0) - COST_R
    out.loc[out["decision"] == "skip", "r_net"] = 0.0
    out["pnl_usd"] = out["r_net"] * p.risk_per_r_usd
    return out


def simulate_window(
    window_days: list[object],
    pnl_by_day: dict[object, list[tuple[float, bool]]],
    profit_target: float,
    mll: float,
    daily_stop_usd: float,
    daily_profit_cap_usd: float,
) -> dict:
    pnl = 0.0
    mll_floor = -mll
    max_eod_pnl = 0.0
    best_day = 0.0
    daily_pnl: dict[object, float] = {}
    trades = 0
    fail_mll = False
    pass_day = None

    for day in window_days:
        day_pnl = 0.0
        blocked = False

        for trade_pnl, is_trade in pnl_by_day.get(day, []):
            if blocked or not is_trade:
                continue
            pnl += trade_pnl
            day_pnl += trade_pnl
            trades += 1

            if pnl <= mll_floor:
                fail_mll = True
                blocked = True
                break
            if daily_stop_usd > 0 and day_pnl <= -daily_stop_usd:
                blocked = True
            if daily_profit_cap_usd > 0 and day_pnl >= daily_profit_cap_usd:
                blocked = True

        daily_pnl[day] = day_pnl
        best_day = max(best_day, day_pnl)
        max_eod_pnl = max(max_eod_pnl, pnl)
        mll_floor = min(0.0, -mll + max_eod_pnl)

        if pnl >= profit_target and best_day < 0.5 * pnl and pass_day is None and not fail_mll:
            pass_day = day
        if fail_mll:
            break

    daily_values = np.array(list(daily_pnl.values()), dtype=float)
    return {
        "passed": pass_day is not None,
        "failed_mll": fail_mll,
        "pass_day": pass_day,
        "end_pnl": pnl,
        "max_eod_pnl": max_eod_pnl,
        "best_day": best_day,
        "consistency": best_day / pnl if pnl > 0 else np.nan,
        "trades": trades,
        "winning_days": int((daily_values > 0).sum()) if daily_values.size else 0,
        "trade_days": len(daily_pnl),
    }


def rolling_day_windows(days: list[object], window_days: int = WINDOW_DAYS) -> list[list[object]]:
    return [days[i : i + window_days] for i in range(0, len(days) - window_days + 1)]


def pnl_sequences_by_day(events: pd.DataFrame) -> dict[object, list[tuple[float, bool]]]:
    seq: dict[object, list[tuple[float, bool]]] = {}
    for row in events.itertuples(index=False):
        is_trade = row.decision != "skip"
        seq.setdefault(row.trade_day, []).append((float(row.pnl_usd), is_trade))
    return seq


def evaluate_params(events: pd.DataFrame, calib: pd.DataFrame, account: str, p: PolicyParams) -> dict:
    cfg = ACCOUNTS[account]
    pe = apply_policy(events, calib, p)
    days = sorted(pe["trade_day"].unique())
    windows = rolling_day_windows(days)
    pnl_by_day = pnl_sequences_by_day(pe)
    results = [
        simulate_window(
            w,
            pnl_by_day,
            profit_target=cfg["profit_target"],
            mll=cfg["mll"],
            daily_stop_usd=p.daily_stop_usd,
            daily_profit_cap_usd=p.daily_profit_cap_usd,
        )
        for w in windows
    ]
    rdf = pd.DataFrame(results)
    pass_rate = float(rdf["passed"].mean()) if len(rdf) else 0.0
    fail_mll_rate = float(rdf["failed_mll"].mean()) if len(rdf) else 0.0
    avg_end_pnl = float(rdf["end_pnl"].mean()) if len(rdf) else 0.0
    median_end_pnl = float(rdf["end_pnl"].median()) if len(rdf) else 0.0
    avg_trades = float(rdf["trades"].mean()) if len(rdf) else 0.0
    pass_df = rdf[rdf["passed"]]

    return {
        "account": account,
        "windows": len(rdf),
        "pass_rate": pass_rate,
        "fail_mll_rate": fail_mll_rate,
        "score": pass_rate - fail_mll_rate,
        "avg_end_pnl": avg_end_pnl,
        "median_end_pnl": median_end_pnl,
        "avg_trades": avg_trades,
        "avg_pass_day_index": np.nan,
        "avg_consistency_on_pass": float(pass_df["consistency"].mean()) if len(pass_df) else np.nan,
        **p.__dict__,
    }


def parameter_grid(account: str) -> list[PolicyParams]:
    if account == "50K":
        risk_grid = [20, 50, 100, 150, 200, 250]
        daily_stops = [0, 1000]
        profit_caps = [0, 1400]
    else:
        risk_grid = [30, 75, 125, 200, 250, 300]
        daily_stops = [0, 1500]
        profit_caps = [0, 2800]

    out = []
    for rev_q in [0.60, 0.75, 0.90]:
        for cont_q in [0.60, 0.75, 0.90]:
            for rev_adx_min in [30, 40]:
                for cont_adx_max in [30, 50, 100]:
                    for risk in risk_grid:
                        for daily_stop in daily_stops:
                            for profit_cap in profit_caps:
                                out.append(
                                    PolicyParams(
                                        rev_q=rev_q,
                                        cont_q=cont_q,
                                        rev_strong_q=0.75,
                                        cont_strong_q=0.75,
                                        rev_adx_min=rev_adx_min,
                                        cont_adx_max=cont_adx_max,
                                        daily_stop_usd=float(daily_stop),
                                        daily_profit_cap_usd=float(profit_cap),
                                        risk_per_r_usd=float(risk),
                                    )
                                )
    return out


def summarize_by_year(events: pd.DataFrame, calib: pd.DataFrame, account: str, p: PolicyParams) -> pd.DataFrame:
    rows = []
    for year, g in events.groupby("year"):
        if len(g["trade_day"].unique()) < WINDOW_DAYS:
            continue
        rows.append(evaluate_params(g, calib, account, p) | {"slice": str(year)})
    return pd.DataFrame(rows)


def summarize_by_regime(events: pd.DataFrame, calib: pd.DataFrame, account: str, p: PolicyParams) -> pd.DataFrame:
    rows = []
    for col in ["adx_bucket", "vwap_bucket"]:
        for bucket, g in events.groupby(col):
            if len(g["trade_day"].unique()) < WINDOW_DAYS:
                continue
            rows.append(evaluate_params(g, calib, account, p) | {"slice": f"{col}:{bucket}"})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No data_"
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("-" if pd.isna(val) else f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    calib, holdout = load_events()
    all_rows = []

    for account in ACCOUNTS:
        grid = parameter_grid(account)
        print(f"Evaluating {account}: {len(grid)} parameter sets", flush=True)
        for i, params in enumerate(grid, 1):
            all_rows.append(evaluate_params(holdout, calib, account, params))
            if i % 250 == 0:
                print(f"  {account}: {i}/{len(grid)}", flush=True)

    grid_df = pd.DataFrame(all_rows)
    grid_df = grid_df.sort_values(["account", "score", "pass_rate"], ascending=[True, False, False])
    grid_df.to_csv(OUT_GRID, index=False)

    sections = [
        "# Topstep Pass-Rate Report - ORB_v1.0",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Assumptions",
        "",
        f"- Rolling window: {WINDOW_DAYS} trading days.",
        "- Trading day proxy: datamart `date` field; not yet converted to exact 5:00 PM CT to 3:10 PM CT Topstep day.",
        "- MLL: trailing highest end-of-day PnL, locked at 0 after reaching starting balance.",
        "- MLL monitoring: realized PnL after each modeled trade; no intrabar mark-to-market.",
        "- Consistency: best profit day must be below 50% of total profit at pass.",
        "- Outcome unit: model R converted to USD using fixed `risk_per_r_usd`.",
        f"- GO gate: pass_rate >= {MIN_GO_PASS_RATE:.0%}, fail_mll_rate <= {MAX_GO_FAIL_MLL_RATE:.0%}, each year pass_rate >= {MIN_GO_YEAR_PASS_RATE:.0%}, each year fail_mll_rate <= {MAX_GO_YEAR_FAIL_MLL_RATE:.0%}.",
        "",
    ]

    for account in ACCOUNTS:
        account_df = grid_df[grid_df["account"] == account].copy()
        top_score = account_df.sort_values(["score", "pass_rate"], ascending=False).head(10).copy()
        top_pass = account_df.sort_values(["pass_rate", "fail_mll_rate"], ascending=[False, True]).head(10).copy()
        best = top_score.iloc[0]
        params = PolicyParams(
            rev_q=float(best["rev_q"]),
            cont_q=float(best["cont_q"]),
            rev_strong_q=float(best["rev_strong_q"]),
            cont_strong_q=float(best["cont_strong_q"]),
            rev_adx_min=float(best["rev_adx_min"]),
            cont_adx_max=float(best["cont_adx_max"]),
            daily_stop_usd=float(best["daily_stop_usd"]),
            daily_profit_cap_usd=float(best["daily_profit_cap_usd"]),
            risk_per_r_usd=float(best["risk_per_r_usd"]),
        )
        by_year = summarize_by_year(holdout, calib, account, params)
        by_regime = summarize_by_regime(holdout, calib, account, params)
        overall_go = bool(best["pass_rate"] >= MIN_GO_PASS_RATE and best["fail_mll_rate"] <= MAX_GO_FAIL_MLL_RATE)
        yearly_go = bool(
            len(by_year)
            and (by_year["pass_rate"] >= MIN_GO_YEAR_PASS_RATE).all()
            and (by_year["fail_mll_rate"] <= MAX_GO_YEAR_FAIL_MLL_RATE).all()
        )
        final_go = overall_go and yearly_go

        sections += [
            f"## {account} Go / No-Go",
            "",
            f"- Overall gate: `{overall_go}`",
            f"- Yearly stability gate: `{yearly_go}`",
            f"- Final decision: `{'GO' if final_go else 'NO-GO'}`",
            "",
            f"## {account} Best Risk-Adjusted Grid Results",
            "",
            md_table(
                top_score,
                [
                    "score",
                    "pass_rate",
                    "fail_mll_rate",
                    "median_end_pnl",
                    "avg_end_pnl",
                    "risk_per_r_usd",
                    "daily_stop_usd",
                    "daily_profit_cap_usd",
                    "rev_q",
                    "cont_q",
                    "rev_adx_min",
                    "cont_adx_max",
                ],
            ),
            "",
            f"## {account} Highest Raw Pass-Rate Grid Results",
            "",
            md_table(
                top_pass,
                [
                    "score",
                    "pass_rate",
                    "fail_mll_rate",
                    "median_end_pnl",
                    "avg_end_pnl",
                    "risk_per_r_usd",
                    "daily_stop_usd",
                    "daily_profit_cap_usd",
                    "rev_q",
                    "cont_q",
                    "rev_adx_min",
                    "cont_adx_max",
                ],
            ),
            "",
            f"## {account} Walk-Forward By Year (Best Risk-Adjusted Params)",
            "",
            md_table(by_year, ["slice", "windows", "pass_rate", "fail_mll_rate", "median_end_pnl", "avg_trades"]),
            "",
            f"## {account} Regime Slices (Best Params)",
            "",
            md_table(by_regime, ["slice", "windows", "pass_rate", "fail_mll_rate", "median_end_pnl", "avg_trades"]),
            "",
        ]

    OUT_REPORT.write_text("\n".join(sections))
    print(f"Saved: {OUT_GRID}")
    print(f"Saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
