# MNQ ML Research

Research namespace for Micro Nasdaq futures experiments.

This tree is intentionally separate from `pipeline/super_structure_ml`, which is
the historical Gold/MGC Super Structure namespace. Do not wire anything here to
`pipeline/live` until a model passes its own data gate, walk-forward review, and
live-risk review.

## Layout

```text
pipeline/mnq_ml/
└── experiments/
    ├── orb_vol_target/
    └── m1_pullback_scalper/
```

Expected outputs:

```text
data/Level_1_Features/mnq/
data/Level_2_Datamart/mnq/
model/MNQ/
```

The existing `model/SUPER_STRUCTURE` and `data/.../super_structure_ml` paths are
legacy Gold/MGC paths and should not be reused for MNQ.

## Active Experiment: `orb_vol_target`

Current state:

```text
Data grain: M1
Primary scoreboard: Topstep-style 30 calendar day window
Current best candidate: 15m OR, long only, TP 2R or 15:00 NY, risk $500
Sweep worker: pipeline/mnq_ml/experiments/orb_vol_target/sweep_orb_params.py
Sweep results: data/Level_2_Datamart/mnq/orb_vol_target/sweeps/sweep_results.parquet
Risk-adjusted model: pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/
Current V2 model: model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_report.md
```

The baseline `events.parquet` is still the simple long-only time-exit build.
Parameter selection should be read from the sweep artifacts, not from the
baseline event file alone.

## L0 Updates

```bash
python3 pipeline/mnq_ml/build_l0_duckdb_from_databento.py --force
python3 pipeline/mnq_ml/fetch_yfinance_1m.py
python3 pipeline/mnq_ml/fetch_yfinance_daily_confluence.py
python3 pipeline/mnq_ml/build_derived_timeframes.py --force
python3 pipeline/mnq_ml/audit_l0_duckdb.py
python3 pipeline/mnq_ml/audit_yfinance_timeframe_parity.py
```

Databento remains the historical source. Yahoo Finance is only a short-window
1m bridge for recent MNQ bars and is stored as `source_symbol='MNQ=F_YF'`.

Daily Yahoo Finance confluence data is stored separately in:

```text
data/Level_0_Raw/yfinance_daily.duckdb
table: daily_ohlcv
symbols: SPY, QQQ, VIX, TNX, DXY
```

This daily data is for regime/context features only. For MNQ trade date `D`,
feature builders must use external daily rows with `date <= D-1`.

## Active Risk-Adjusted Model

The current trainable probability dataset is:

```text
data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/breakout_quality_features.parquet
rows: 2,559
columns: 77
features: 62
```

V2 confluence commands:

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/build_daily_confluence_features.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/audit_daily_confluence_features.py
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/train_risk_adjusted_v2.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/simulate_kelly_overlay_v2.py --force
```

V2 readout:

- `success_2r` logistic holdout AUC is roughly flat versus V1.
- `success_2r` LightGBM improves versus V1, but remains weaker by ranking.
- Kelly V2 holdout improves for normalized overlays, but latest 5D/10D windows
  are negative.
- Do not wire this to live. Next step is Topstep MLL/consistency simulation and
  calibration review.
