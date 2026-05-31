# Research Program

Last updated: 2026-05-30

## Active Program: MNQ ORB Topstep Research

The active research program is MNQ only. MGC/Super Structure documents remain in
the repo as historical context and live-system context, but the current research
loop below does not use MGC artifacts.

```text
Instrument: MNQ / Micro Nasdaq
Primary source: data/Level_0_Raw/MNQ_1m.duckdb
Research code: pipeline/mnq_ml/experiments/orb_vol_target/
L1: data/Level_1_Features/mnq/orb_vol_target/
L2: data/Level_2_Datamart/mnq/orb_vol_target/
Sweep outputs: data/Level_2_Datamart/mnq/orb_vol_target/sweeps/
Risk-adjusted ML: pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/
Model outputs: model/MNQ/orb_vol_target/orb_risk_adjusted_model/
```

## Karpathy-Style Iteration Loop

Use this loop for each research pass:

1. Establish the simplest baseline.
2. Measure it with the actual scoreboard.
3. Inspect the failure mode.
4. Change one core idea.
5. Re-measure with the same scoreboard.
6. Keep the change only if the evidence survives short-window and robustness checks.

For MNQ ORB the scoreboard is Topstep-aware:

```text
30D PnL target: >= $3,000
30D drawdown: must stay materially below the 50K MLL
Trade count: enough activity for a 30-day evaluation
Consistency: best-day concentration must be checked before promotion
Robustness: 50D/100D/200D and year-by-year behavior must not collapse
```

## Current Iteration State

### Iteration 0: M1 Time-Exit Baseline

Hypothesis:

```text
Opening range breakout on MNQ has continuation edge after NY cash open.
```

Baseline:

```text
OR: 30m
Side: long only
Entry: first M1 close above OR high, enter next M1 open
Exit: 15:00 NY
Sizing: fixed target risk using opposite OR as risk reference
```

Result:

```text
Risk $200 was too conservative for 2026 volatility.
Many valid setups were rejected because integer sizing produced < 1 contract.
```

Decision:

```text
Keep M1 as the baseline data grain.
Do not treat $200 as the final risk setting.
```

### Iteration 1: OR Duration / Side / Risk Sweep

Grid:

```text
orb_minutes: 10, 15, 20, 30
side_mode: long, short, long_short
target_risk_usd: 100, 200, 300, 400, 500, 600
exit_mode: time_exit
```

Finding:

```text
Long-only was cleaner than short or long_short on full history.
30m long risk $600 had strong full-history Return/DD, but recent 30D was weak.
15m long started to look better on the Topstep 30D lens.
```

Decision:

```text
Do not promote long_short yet.
Keep long-only as the primary candidate path.
```

### Iteration 2: TP 2R Or Time Exit

Change:

```text
If trade reaches +2R before 15:00 NY, close immediately.
Otherwise close at 15:00 NY.
```

Best current candidate:

```text
OR: 15m
Side: long only
Exit: TP 2R or 15:00 NY
Risk: $500
```

Current evidence:

| Window | Trades | PnL | Max DD |
| ---: | ---: | ---: | ---: |
| 30D | 18 | $3,491 | -$549 |
| 50D | 30 | $5,448 | -$859 |
| 100D | 54 | $4,135 | -$4,066 |
| 200D | 94 | $5,569 | -$4,556 |

Decision:

```text
Candidate is worth deeper Topstep evaluation.
It is not live-ready until consistency and MLL simulations pass.
```

### Iteration 3: Daily Confluence V2 Probability Model

Change:

```text
Add prior daily SPY, QQQ, VIX, TNX, and DXY context as a no-lookahead feature
family. Train V2 probability models on 62 features.
```

Current dataset:

```text
rows: 2,559
columns: 77
features: 62
holdout: 2026-01-02 through 2026-05-27
```

V2 model readout:

| Target | Model | Holdout AUC | Holdout PR-AUC | Holdout Brier | Delta AUC vs V1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `success_2r` | logistic | 0.716 | 0.364 | 0.212 | -0.001 |
| `success_2r` | lgbm_shallow | 0.633 | 0.236 | 0.138 | +0.027 |
| `positive_eod` | logistic | 0.584 | 0.641 | 0.246 | +0.042 |
| `positive_eod` | lgbm_shallow | 0.515 | 0.588 | 0.245 | -0.034 |

V2 Kelly holdout, integer executable:

| Variant | PnL | Max DD | Return/DD |
| --- | ---: | ---: | ---: |
| `fixed_1.00x` | $17,140 | -$3,430 | 5.00 |
| `basefloor_kelly_0.10x` | $18,076 | -$3,436 | 5.26 |
| `norm_target_600` | $21,676 | -$4,954 | 4.38 |
| `norm_target_750` | $27,857 | -$6,220 | 4.48 |

Latest V2 30D window remains weak:

| Variant | 30D Trades | 30D PnL | 30D Max DD |
| --- | ---: | ---: | ---: |
| `fixed_1.00x` | 45 | $1,423 | -$3,430 |
| `basefloor_kelly_0.10x` | 45 | $1,680 | -$3,436 |
| `norm_target_750` | 45 | $2,778 | -$4,034 |

Decision:

```text
Keep V2 confluence as the active benchmark, but do not promote it.
The next iteration should measure Topstep MLL/consistency and calibration
failure modes before adding HMM.
```

## Current P0/P1/P2

### P0

- Add Topstep-style MLL and consistency simulator for the V2 risk-adjusted
  event stream.
- Add calibration table by probability decile and side.
- Decide whether HMM should target trend/vol regime or whether the failure is
  mainly position sizing and recent-window drawdown.

### P1

- Add HMM only as a no-lookahead feature family after P0 simulator explains the
  failure mode.
- Add year-by-year and regime breakdown for the selected candidate.
- Confirm slippage/commission assumptions for MNQ Topstep fills.
- Add sensitivity around TP: 1.5R, 2R, 2.5R, 3R.
- Add time exits: 14:00, 15:00, 15:30, 16:00 NY.

### P2

- Only if P0/P1 survive: build a trainable feature layer or simple filter.
- Candidate feature families: OR range, breakout time, volume, distance from OR,
  prior session trend, volatility regime.
- No live wiring before a promotion report exists.
