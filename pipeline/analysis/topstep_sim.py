"""
Topstep Trading Combine simulator — shared module.

Refinements over inline implementations in objective_sweep_orb_v*.py:

1. **Trading day boundaries (eksak)**
   - Topstep trading day: 5:00 PM CT to 3:10 PM CT next day
   - Uses `America/Chicago` timezone to map breakout_ts to correct trade_day
   - Tokyo session (00:00-03:00 UTC) maps to PREVIOUS calendar day's trading day
   - US session (13:30-16:30 UTC) maps to CURRENT calendar day's trading day
   - Properly groups all three sessions (Tokyo→London→US) into same trading day

2. **Komisi riil**
   - Fixed $3.00 per round turn (commission + slippage) for MGC Micro Gold Futures
   - Not percentage-of-risk like previous 0.07R
   - MGC: 1 tick = $0.10/oz = $1.00/contract, commission ~$2.00 round turn, slippage ~$1.00

3. **Consistency rule**
   - Best day < 50% of total profit at pass time
   - Only counts profitable days (not losing days)
   - Checked continuously as PnL evolves

4. **MLL trailing**
   - Trailing highest end-of-day PnL
   - Locked at starting balance (can't go below initial MLL)
   - Same as previous implementation (was correct)

Usage:
    from topstep_sim import (
        map_to_topstep_trade_day,
        simulate_window,
        score_policy_on_events,
        COMMISSION_PER_TRADE,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ── MGC contract specs ──────────────────────────────────────────────────────
# MGC = 10 troy oz Micro Gold Futures
# 1 tick = $0.10/oz = $1.00/contract
# 1 point = $10.00/contract

COMMISSION_PER_TRADE = 3.00  # $3.00 round turn (entry + exit + slippage)

ACCOUNTS = {
    "50K": {"profit_target": 3000.0, "mll": 2000.0},
    "100K": {"profit_target": 6000.0, "mll": 3000.0},
}

MIN_GO_PASS_RATE = 0.60
MAX_GO_FAIL_MLL_RATE = 0.10
MIN_GO_YEAR_PASS_RATE = 0.30
MAX_GO_YEAR_FAIL_MLL_RATE = 0.20


# ── trading day mapping ─────────────────────────────────────────────────────

def map_to_topstep_trade_day(ts_series: pd.Series) -> pd.Series:
    """
    Map UTC timestamps to Topstep trading days.

    Topstep trading day: 5:00 PM CT to 3:10 PM CT next day.
    Equivalent: trading_day = calendar_date_CT of (timestamp_ct - 15h10m).

    Args:
        ts_series: Series of UTC timestamps (datetime64[ns, UTC])

    Returns:
        Series of date objects representing the Topstep trading day.
    """
    # Convert to America/Chicago
    ts_ct = ts_series.dt.tz_convert("America/Chicago")
    # Shift: if after 3:10 PM CT, it belongs to NEXT trading day
    # Equivalent: subtract 15h10m, take the date
    offset = pd.Timedelta(hours=15, minutes=10)
    trading_day = (ts_ct - offset).dt.date
    return trading_day


def map_to_topstep_trade_day_dm(dm: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'trade_day' column to datamart using Topstep trading day boundaries.
    Also adds 'trade_day_dt' as a proper datetime for grouping.
    Modifies in-place and returns the dataframe.
    """
    dm["trade_day"] = map_to_topstep_trade_day(dm["breakout_ts"])
    dm["trade_day_dt"] = pd.to_datetime(dm["trade_day"])
    return dm


# ── policy application ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyParams:
    rev_q: float
    cont_q: float
    rev_adx_min: float
    cont_adx_max: float
    daily_stop_usd: float
    daily_profit_cap_usd: float
    risk_per_r_usd: float


def apply_policy(
    events: pd.DataFrame,
    calib: pd.DataFrame,
    p: PolicyParams,
    rr: float,
) -> pd.DataFrame:
    """
    Apply dynamic rev/cont/skip policy to events.

    Returns events with added columns: decision, y, r_net, pnl_usd
    """
    out = events.copy()
    rev_t = float(np.percentile(calib["prob_rev"], p.rev_q * 100))
    cont_t = float(np.percentile(calib["prob_cont"], p.cont_q * 100))
    rev_strong = float(np.percentile(calib["prob_rev"], 75))
    cont_strong = float(np.percentile(calib["prob_cont"], 75))

    trend_sign = np.sign(out["ema_slope_1h"].fillna(0))
    aligned_rev = trend_sign == -out["breakout_side"]
    aligned_cont = trend_sign == out["breakout_side"]

    rev_gate = (
        (out["prob_rev"] >= rev_t) & aligned_rev &
        ((out["adx_14_15m"] >= p.rev_adx_min) | (out["prob_rev"] >= rev_strong))
    )
    cont_gate = (
        (out["prob_cont"] >= cont_t) & aligned_cont &
        ((out["adx_14_15m"] <= p.cont_adx_max) | (out["prob_cont"] >= cont_strong))
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
    out["y"] = np.where(
        out["decision"] == "rev", out["y_rev"],
        np.where(out["decision"] == "cont", out["y_cont"], np.nan)
    )

    # PnL calculation with fixed commission
    gross_pnl = np.where(out["y"] == 1, rr, -1.0) * p.risk_per_r_usd
    out["pnl_usd"] = np.where(out["decision"] == "skip", 0.0, gross_pnl - COMMISSION_PER_TRADE)

    return out


# ── window simulation ───────────────────────────────────────────────────────

def simulate_window(
    window_days: list,
    pnl_by_day: dict,
    profit_target: float,
    mll: float,
    daily_stop_usd: float = 0.0,
    daily_profit_cap_usd: float = 0.0,
) -> dict:
    """
    Simulate a single 20-trading-day Topstep window.

    Args:
        window_days: ordered list of trading days in the window
        pnl_by_day: dict of {trade_day: [(pnl_usd, is_trade), ...]}
        profit_target: $3,000 for 50K
        mll: $2,000 for 50K
        daily_stop_usd: 0 = no daily stop
        daily_profit_cap_usd: 0 = no daily profit cap

    Returns:
        dict with keys: passed, failed_mll, end_pnl, best_day, trades, etc.
    """
    pnl = 0.0
    mll_floor = -mll
    max_eod_pnl = 0.0
    best_day = 0.0
    trades = 0
    fail_mll = False
    passed = False

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

        best_day = max(best_day, day_pnl)
        max_eod_pnl = max(max_eod_pnl, pnl)
        mll_floor = min(0.0, -mll + max_eod_pnl)

        # Check pass: profit target met AND consistency rule
        if not passed and not fail_mll and pnl >= profit_target:
            # Consistency: best day must be < 50% of total profit
            if best_day < 0.5 * pnl:
                passed = True

        if fail_mll:
            break

    return {
        "passed": passed,
        "failed_mll": fail_mll,
        "end_pnl": pnl,
        "best_day": best_day,
        "trades": trades,
    }


def build_pnl_by_day(events: pd.DataFrame) -> dict:
    """Build {trade_day: [(pnl_usd, is_trade), ...]} from events."""
    pnl_by_day: dict = {}
    for row in events.itertuples(index=False):
        is_trade = row.decision != "skip"
        pnl_by_day.setdefault(row.trade_day, []).append((float(row.pnl_usd), is_trade))
    return pnl_by_day


def build_windows(trade_days: list, window_size: int = 20) -> list[list]:
    """Build rolling windows from sorted unique trade days."""
    return [trade_days[i: i + window_size]
            for i in range(len(trade_days) - window_size + 1)]


# ── full evaluation ─────────────────────────────────────────────────────────

def score_policy_on_events(
    events: pd.DataFrame,
    calib: pd.DataFrame,
    p: PolicyParams,
    rr: float,
    profit_target: float = 3000.0,
    mll: float = 2000.0,
    window_size: int = 20,
) -> dict:
    """
    Full evaluation: apply policy, run Topstep simulator, return metrics.

    Args:
        events: holdout events with prob_rev, prob_cont columns
        calib: calibration events for threshold setting
        p: policy parameters
        rr: risk/reward ratio for the target
        profit_target: Topstep profit target
        mll: maximum loss limit
        window_size: rolling window size in trading days

    Returns:
        dict with pass_rate, fail_mll_rate, score, median_end_pnl, etc.
    """
    pe = apply_policy(events, calib, p, rr)
    days = sorted(pe["trade_day"].unique())
    windows = build_windows(days, window_size)
    if not windows:
        return {"pass_rate": 0.0, "fail_mll_rate": 0.0, "score": 0.0,
                "median_end_pnl": np.nan, "avg_trades": 0.0, "windows": 0}

    pnl_by_day = build_pnl_by_day(pe)
    results = [
        simulate_window(w, pnl_by_day, profit_target, mll,
                        daily_stop_usd=p.daily_stop_usd,
                        daily_profit_cap_usd=p.daily_profit_cap_usd)
        for w in windows
    ]
    rdf = pd.DataFrame(results)
    pass_rate = float(rdf["passed"].mean())
    fail_mll_rate = float(rdf["failed_mll"].mean())
    return {
        "pass_rate": pass_rate,
        "fail_mll_rate": fail_mll_rate,
        "score": pass_rate - fail_mll_rate,
        "median_end_pnl": float(rdf["end_pnl"].median()),
        "avg_trades": float(rdf["trades"].mean()),
        "windows": len(rdf),
    }


def by_year_eval(
    events: pd.DataFrame,
    calib: pd.DataFrame,
    p: PolicyParams,
    rr: float,
    profit_target: float = 3000.0,
    mll: float = 2000.0,
    window_size: int = 20,
) -> dict[str, dict]:
    """
    Evaluate policy per-year on holdout events.
    Returns {year_str: {pass_rate, fail_mll_rate, median_end_pnl}}
    """
    pe = apply_policy(events, calib, p, rr)
    year_results = {}
    for yr, g in pe.groupby("year"):
        days = sorted(g["trade_day"].unique())
        if len(days) < window_size:
            continue
        windows = build_windows(days, window_size)
        pnl_by_day = build_pnl_by_day(g)
        sims = [simulate_window(w, pnl_by_day, profit_target, mll,
                                daily_stop_usd=p.daily_stop_usd,
                                daily_profit_cap_usd=p.daily_profit_cap_usd)
                for w in windows]
        sdf = pd.DataFrame(sims)
        year_results[str(yr)] = {
            "pass_rate": float(sdf["passed"].mean()),
            "fail_mll_rate": float(sdf["failed_mll"].mean()),
            "median_end_pnl": float(sdf["end_pnl"].median()),
        }
    return year_results


# ── GO/NO-GO decision ──────────────────────────────────────────────────────

def go_no_go(
    pass_rate: float,
    fail_mll_rate: float,
    by_year: dict[str, dict],
) -> tuple[bool, bool, bool]:
    """
    Evaluate GO/NO-GO gates.

    Returns:
        (overall_gate, yearly_stability_gate, final_decision)
    """
    overall_go = bool(pass_rate >= MIN_GO_PASS_RATE and fail_mll_rate <= MAX_GO_FAIL_MLL_RATE)
    yearly_go = True
    for yr_d in by_year.values():
        if yr_d["pass_rate"] < MIN_GO_YEAR_PASS_RATE:
            yearly_go = False
        if yr_d["fail_mll_rate"] > MAX_GO_YEAR_FAIL_MLL_RATE:
            yearly_go = False
    return overall_go, yearly_go, (overall_go and yearly_go)


# ── param grid helpers ──────────────────────────────────────────────────────

def build_param_grid() -> list[PolicyParams]:
    """Standard 128-combination grid for objective sweeping."""
    params = []
    for rev_q in [0.60, 0.75]:
        for cont_q in [0.60, 0.75]:
            for rev_adx_min in [30, 40]:
                for cont_adx_max in [30, 100]:
                    for risk in [100, 150, 200, 250]:
                        for profit_cap in [0.0, 1400.0]:
                            params.append(PolicyParams(
                                rev_q=rev_q, cont_q=cont_q,
                                rev_adx_min=rev_adx_min, cont_adx_max=cont_adx_max,
                                daily_stop_usd=0.0, daily_profit_cap_usd=profit_cap,
                                risk_per_r_usd=float(risk),
                            ))
    return params


# ── report helpers ──────────────────────────────────────────────────────────

def fmt(v) -> str:
    if pd.isna(v):
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Convert DataFrame to markdown table."""
    if df.empty:
        return "_No data_"
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = [fmt(row[c]) if c in row.index else "-" for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ── comparison test ─────────────────────────────────────────────────────────

def compare_simulators(
    events: pd.DataFrame,
    calib: pd.DataFrame,
    p: PolicyParams,
    rr: float,
) -> dict:
    """
    Compare old vs new simulator on the same events.

    Old simulator: uses date-based trade_day, 0.07R commission
    New simulator: uses Topstep CT-based trade_day, $3.00 fixed commission

    Returns dict with both results for comparison.
    """
    from datetime import date

    # ── New simulator (Topstep trade days, fixed commission) ──
    new_result = score_policy_on_events(events, calib, p, rr)

    # ── Old simulator (calendar date trade days, 0.07R commission) ──
    old_cost_r = 0.07

    # Make a copy with old-style trade_day
    old_events = events.copy()
    old_events["trade_day"] = pd.to_datetime(old_events["date"]).dt.date

    # Reapply policy with old commission
    old_policy = apply_policy(old_events, calib, p, rr)
    # override PnL with old commission formula
    old_policy["pnl_usd"] = np.where(
        old_policy["decision"] == "skip", 0.0,
        np.where(old_policy["y"] == 1, rr, -1.0) * p.risk_per_r_usd - old_cost_r * p.risk_per_r_usd
    )

    old_days = sorted(old_events["trade_day"].unique())
    old_windows = build_windows(old_days)
    old_pnl_by_day = build_pnl_by_day(old_policy)
    old_results = [
        simulate_window(w, old_pnl_by_day, 3000.0, 2000.0,
                        daily_stop_usd=p.daily_stop_usd,
                        daily_profit_cap_usd=p.daily_profit_cap_usd)
        for w in old_windows
    ]
    old_rdf = pd.DataFrame(old_results)
    old_pass_rate = float(old_rdf["passed"].mean())
    old_fail_mll = float(old_rdf["failed_mll"].mean())

    return {
        "new_pass_rate": new_result["pass_rate"],
        "new_fail_mll": new_result["fail_mll_rate"],
        "new_score": new_result["score"],
        "new_median_pnl": new_result["median_end_pnl"],
        "new_windows": new_result["windows"],
        "old_pass_rate": old_pass_rate,
        "old_fail_mll": old_fail_mll,
        "old_score": old_pass_rate - old_fail_mll,
        "delta_pass_rate": new_result["pass_rate"] - old_pass_rate,
        "delta_score": (new_result["pass_rate"] - new_result["fail_mll_rate"]) - (old_pass_rate - old_fail_mll),
    }
