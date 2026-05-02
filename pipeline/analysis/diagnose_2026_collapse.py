"""
Diagnose 2026 Collapse — understand why ALL targets fail in 2026.

Analyses:
1. ADX distribution by year (2024 vs 2025 vs 2026)
2. Volatility regime (ATR14) by year
3. VWAP position distribution
4. Win rates by ADX bucket × year — reversal vs continuation
5. Trade frequency decline in 2026
6. Label hit rates (all targets) by year
7. Feature drift

Output: model/ORB_v1.0/DIAGNOSE_2026.md
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DM_PATH = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"
OUT_DIR = ROOT / "model/ORB_v1.0"
OUT_PATH = OUT_DIR / "DIAGNOSE_2026.md"

# ── helpers ───────────────────────────────────────────────────────────────────

ADX_BINS = [0, 20, 30, 50, 100]
ADX_LABELS = ["<20", "20-30", "30-50", ">50"]


def adx_bucket(adx: pd.Series) -> pd.Series:
    return pd.cut(adx, bins=ADX_BINS, labels=ADX_LABELS, right=False)


VWAP_BINS = [-float("inf"), -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, float("inf")]
VWAP_LABELS = ["<-1%", "-1% to -0.5%", "-0.5% to -0.2%", "-0.2% to 0%",
               "0% to 0.2%", "0.2% to 0.5%", "0.5% to 1%", ">1%"]


def df_to_md_table(df: pd.DataFrame, col_fmt: dict | None = None) -> str:
    """Convert DataFrame to markdown table without tabulate dependency."""
    rows = []
    # header
    header = "| " + " | ".join(["", *[str(c) for c in df.columns]]) + " |"
    sep = "|" + "|".join(["---"] * (len(df.columns) + 1)) + "|"
    rows.append(header)
    rows.append(sep)

    for idx, row in df.iterrows():
        vals = []
        for col in df.columns:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        line = "| " + " | ".join([str(idx), *vals]) + " |"
        rows.append(line)
    return "\n".join(rows)


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


# ── load ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading datamart...")
    dm = pd.read_parquet(DM_PATH)
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year.astype(int)

    # only interested in 2020+ for recent regime context
    recent = dm[dm["year"] >= 2020].copy()

    # ── 1. Event Count ────────────────────────────────────────────
    print("1. Event counts by year (2020+)...")
    counts = recent.groupby("year").agg(
        n_events=("date", "count"),
        n_days=("date", lambda x: x.nunique()),
    )
    counts["events_per_day"] = counts["n_events"] / counts["n_days"]

    yc = recent.groupby(["year", "side"]).size().unstack(fill_value=0)

    # ── 2. ADX Distribution by Year ───────────────────────────────
    print("2. ADX distribution by year...")
    recent["adx_bucket"] = adx_bucket(recent["adx_14_15m"])
    adx_dist = recent.groupby(["year", "adx_bucket"], observed=True).size().unstack(fill_value=0)
    adx_dist_pct = adx_dist.div(adx_dist.sum(axis=1), axis=0) * 100

    adx_stats = recent.groupby("year")["adx_14_15m"].agg(["mean", "std", "count"])

    # ADX percentiles
    adx_pct = recent.groupby("year")["adx_14_15m"].quantile([0.25, 0.5, 0.75]).unstack()

    # ── 3. Volatility (ATR14) by Year ─────────────────────────────
    print("3. ATR volatility by year...")
    atr_stats = recent.groupby("year")["atr14_at_entry"].agg(["mean", "median", "std", "count"])
    atr_pct = recent.groupby("year")["atr14_at_entry"].quantile([0.25, 0.5, 0.75]).unstack()

    # ── 4. VWAP position by year ──────────────────────────────────
    print("4. VWAP position by year...")
    recent["vwap_bucket"] = pd.cut(recent["price_vs_vwap_pct"], bins=VWAP_BINS, labels=VWAP_LABELS)
    vwap_dist = recent.groupby(["year", "vwap_bucket"], observed=True).size().unstack(fill_value=0)
    vwap_dist_pct = vwap_dist.div(vwap_dist.sum(axis=1), axis=0) * 100

    vwap_stats = recent.groupby("year")["price_vs_vwap_pct"].agg(["mean", "std"])

    # ── 5. Win Rates by ADX bucket × Year (rev side, y_1r4_240m) ─
    print("5. Win rates by ADX bucket...")
    label = "y_1r4_240m"

    def wr_by_adx(df: pd.DataFrame, side: str) -> pd.DataFrame:
        s = df[df["side"] == side].copy()
        s["adx_bucket"] = adx_bucket(s["adx_14_15m"])
        g = s.groupby(["year", "adx_bucket"], observed=True)
        result = g[label].agg(["mean", "count"])
        result["mean"] = result["mean"] * 100
        return result

    wr_rev = wr_by_adx(recent, "rev")
    wr_cont = wr_by_adx(recent, "cont")

    # ── 6. Label hit rates by year (all targets) ──────────────────
    print("6. Label hit rates by year for all targets...")
    label_cols = [c for c in dm.columns if c.startswith("y_")]
    yr_labels = recent.groupby("year")[label_cols].mean() * 100

    def label_hit_by_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
        s = df[df["side"] == side]
        return s.groupby("year")[label_cols].mean() * 100

    rev_labels = label_hit_by_side(recent, "rev")
    cont_labels = label_hit_by_side(recent, "cont")

    # ── 7. Feature Drift ──────────────────────────────────────────
    print("7. Feature drift...")
    feature_cols = [
        "orb_range_atr_ratio", "breakout_strength", "atr14_at_entry",
        "price_vs_vwap_pct", "adx_14_15m", "time_in_session_min",
        "orb_range",
    ]
    feat_stats = recent.groupby("year")[feature_cols].agg(["mean", "std", "median"])

    # ── 8. Trade frequency by year (unfiltered) ───────────────────
    print("8. Trade frequency by year...")
    # Days with events per year
    days_with_events = recent.groupby("year")["date"].nunique().to_frame("trading_days")

    # ── output ────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def h1(t: str) -> None:
        lines.append(f"\n# {t}\n")

    def h2(t: str) -> None:
        lines.append(f"\n## {t}\n")

    def table(df: pd.DataFrame, caption: str = "") -> None:
        if caption:
            lines.append(f"**{caption}**\n")
        lines.append(df_to_md_table(df))
        lines.append("")

    lines.append("# 2026 Collapse Diagnosis — ORB Strategy")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\nFocus: understand why all 10 label targets achieve **0% pass_rate in 2026**.")

    h1("1. Event Volume by Year (2020+)")
    table(counts, "Events, trading days, and daily frequency")
    table(yc, "Events by side (rev/cont)")

    h1("2. ADX Distribution")
    table(adx_stats, "ADX mean, std, count by year")
    table(adx_pct, "ADX percentiles (25/50/75) by year")
    lines.append("**ADX bucket distribution (% of events):**\n")
    lines.append(df_to_md_table(adx_dist_pct))
    lines.append("")

    h1("3. Volatility Regime (ATR14)")
    table(atr_stats, "ATR14 at entry — mean, median, std")
    table(atr_pct, "ATR14 percentiles (25/50/75) by year")

    h1("4. VWAP Position")
    table(vwap_stats, "VWAP distance % — mean and std")
    lines.append("**VWAP position distribution (% of events):**\n")
    lines.append(df_to_md_table(vwap_dist_pct))
    lines.append("")

    h1(f"5. Win Rate {label} by ADX Bucket")
    lines.append("### Reversal:\n")
    lines.append(df_to_md_table(wr_rev))
    lines.append("")
    lines.append("### Continuation:\n")
    lines.append(df_to_md_table(wr_cont))
    lines.append("")

    h1("6. All Label Hit Rates (%)")
    table(yr_labels, "All sides combined")
    lines.append("### Reversal only:\n")
    lines.append(df_to_md_table(rev_labels))
    lines.append("")
    lines.append("### Continuation only:\n")
    lines.append(df_to_md_table(cont_labels))
    lines.append("")

    h1("7. Feature Drift (Mean by Year)")
    table(feat_stats, "Feature means, stds, medians")

    h1("8. Trading Days by Year")
    table(days_with_events)

    # ── Key observations ──────────────────────────────────────────
    h1("Key Observations")

    # Pull key numbers
    def safe_val(df, year, col):
        if year in df.index:
            return df.loc[year, col]
        return None

    for yr_label, yr_val in [("2024", 2024), ("2025", 2025), ("2026", 2026)]:
        n = safe_val(counts, yr_val, "n_events")
        d = safe_val(counts, yr_val, "n_days")
        epd = safe_val(counts, yr_val, "events_per_day")
        adx_m = safe_val(adx_stats, yr_val, "mean")
        atr_m = safe_val(atr_stats, yr_val, "mean")
        lines.append(f"\n### {yr_label}")
        lines.append(f"- Events: {n} over {d} days ({epd:.1f}/day)")
        lines.append(f"- ADX mean: {adx_m:.1f}")
        lines.append(f"- ATR14 mean: {atr_m:.2f}")

    # Best label WR by year
    lines.append(f"\n### Best Target ({label}) Win Rates by Year")
    for yr_val in [2024, 2025, 2026]:
        wr_r = safe_val(rev_labels, yr_val, label)
        wr_c = safe_val(cont_labels, yr_val, label)
        lines.append(f"- {yr_val}: rev WR={wr_r:.1f}%, cont WR={wr_c:.1f}%")

    # ADX bucket shifts
    for yr_val in [2024, 2025, 2026]:
        if yr_val in adx_dist_pct.index:
            pct_under20 = adx_dist_pct.loc[yr_val, "<20"] if "<20" in adx_dist_pct.columns else 0
            pct_over50 = adx_dist_pct.loc[yr_val, ">50"] if ">50" in adx_dist_pct.columns else 0
            lines.append(f"- {yr_val}: ADX<20={pct_under20:.1f}%, ADX>50={pct_over50:.1f}%")

    # Hypothesis section
    lines.append("""

## Hypothesis Evaluation

Based on the data above, diagnose which hypothesis explains the 2026 collapse:

### Hypothesis A — ADX Regime Shift ❓
If ADX distribution shifted significantly toward <20 in 2026, the ADX gate loses discriminative power.

### Hypothesis B — Volatility Regime Change ❓
ATR14 doubling or halving would directly impact SL width and TP attainability (4R requires price to travel 6× ATR).

### Hypothesis C — VWAP Position Shift ❓
If price consistently stays on one VWAP side, VWAP-based features become useless.

### Hypothesis D — Trade Starvation ❓
Fewer breakout events in 2026 → fewer trades → harder to accumulate $3,000 in 20 days.

### Hypothesis E — ORB Structure Broken ❓
If ALL label win rates collapsed uniformly regardless of side/horizon/RR, then the ORB breakout pattern itself stopped working in 2026.

### Hypothesis F — Label Distribution Collapse ❓
If the variance of label outcomes (y) dropped to near-zero, the model cannot discriminate at all.

---

## Conclusion

See the observations above to determine which hypothesis(es) hold.
""")

    OUT_PATH.write_text("\n".join(lines))
    print(f"\n✅ Report saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
