# MNQ Models

Model artifacts for Micro Nasdaq futures research.

This namespace is separate from `model/SUPER_STRUCTURE`, which remains the
legacy Gold/MGC Super Structure model namespace.

No MNQ model should be considered live-ready unless its experiment gate report
shows `training_allowed: true` and its walk-forward report passes promotion
requirements.

## Current Model Status

There is no promoted MNQ live model yet.

Start here when asking "model mana yang dipakai":

```text
model/MNQ/ORB/
```

Current named baseline candidate:

```text
rule_based_15m_long_tp2r_eod
```

Contract:

```text
15m OR, long only, first M1 close above OR high,
entry next M1 open, TP 2R or 15:00 NY, risk $500,
no normal strategic SL.
```

Canonical data artifacts:

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/
```

Current net-of-cost snapshot:

```text
PnL $33,091 / DD -$12,124 / PF 1.12 / Sharpe 0.50 / Sortino 0.64
```

This is not live-ready. Promotion still requires Topstep consistency, MLL, and
walk-forward review.

Active ML research namespace:

```text
model/MNQ/ORB/orb_risk_adjusted_model/
```

This is not the current baseline engine. It is a probability/risk-sizing
research overlay that must beat the unchanged rule-based candidate before it is
promoted.

Current model-ready dataset:

```text
data/Level_2_Datamart/mnq/ORB/orb_risk_adjusted_model/breakout_quality_features.parquet
rows: 2,559
columns: 77
features: 62
```

The 62-feature dictionary is documented in:

```text
model/MNQ/ORB/orb_risk_adjusted_model/README.md
```

V2 confluence artifacts now exist:

```text
risk_adjusted_v2_metrics.json
risk_adjusted_v2_report.md
risk_adjusted_v2_kelly_overlay_report.md
```

Current readout: confluence helps some diagnostics and Kelly overlays, but it is
not live-ready. Latest 5D/10D risk-adjusted windows are negative, so the next
gate is Topstep MLL/consistency simulation and probability calibration before
adding HMM features.
