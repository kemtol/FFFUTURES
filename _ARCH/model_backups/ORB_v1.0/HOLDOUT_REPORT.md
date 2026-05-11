# Holdout Evaluation — ORB_v1.0

Generated: 2026-05-06 11:45

## Setup

- Target: `y_1r2_120m` (reversal only model)
- Holdout window: `2024-01-01` onward
- Threshold calibration window: `2022-01-01` to `2023-12-31`
- Selection rule: `prob >= 0.328765` (top 40% equivalent on calibration set)
- Selection share on holdout: `43.21%`
- Transaction cost assumed: `0.07R` per trade

## Headline (Holdout 2024+)

### Reversal (all signals, no model filter)

| bucket | n | win_rate | exp_gross_R | exp_net_R | pf_net | sharpe_trade_net | max_loss_streak |
|---|---|---|---|---|---|---|---|
| rev_all | 5291 | 0.334 | 0.001 | -0.069 | 0.904 | -3.531 | 33 |

### Reversal (model-filtered)

| bucket | n | win_rate | exp_gross_R | exp_net_R | pf_net | sharpe_trade_net | max_loss_streak |
|---|---|---|---|---|---|---|---|
| rev_model_top40 | 2286 | 0.361 | 0.083 | 0.013 | 1.019 | 0.421 | 24 |

### Continuation baseline (all signals)

| bucket | n | win_rate | exp_gross_R | exp_net_R | pf_net | sharpe_trade_net | max_loss_streak |
|---|---|---|---|---|---|---|---|
| cont_all | 5291 | 0.341 | 0.023 | -0.047 | 0.933 | -2.410 | 26 |

## North Star Check (on model-filtered holdout)

- PF net > 1.3: `False`
- Sharpe trade net > 1.0: `False`
- Max loss streak < 8: `False`

## By Year — Reversal

### All reversal signals

| bucket | n | win_rate | exp_net_R | pf_net | max_loss_streak |
|---|---|---|---|---|---|
| 2024 | 2301 | 0.349 | -0.023 | 0.967 | 23 |
| 2025 | 2281 | 0.316 | -0.123 | 0.832 | 33 |
| 2026 | 709 | 0.343 | -0.042 | 0.941 | 18 |

### Model-filtered reversal

| bucket | n | win_rate | exp_net_R | pf_net | max_loss_streak |
|---|---|---|---|---|---|
| 2025 | 1020 | 0.326 | -0.091 | 0.874 | 15 |
| 2024 | 1002 | 0.394 | 0.113 | 1.174 | 24 |
| 2026 | 264 | 0.367 | 0.032 | 1.048 | 15 |

## Regime Breakdown (Model-filtered Reversal)

### ADX bucket

| bucket | n | win_rate | exp_net_R | pf_net | max_loss_streak |
|---|---|---|---|---|---|
| 30-50 | 747 | 0.384 | 0.083 | 1.125 | 17 |
| 20-30 | 729 | 0.332 | -0.074 | 0.896 | 15 |
| <20 | 516 | 0.345 | -0.035 | 0.950 | 18 |
| >50 | 294 | 0.401 | 0.134 | 1.209 | 9 |

### Price vs VWAP

| bucket | n | win_rate | exp_net_R | pf_net | max_loss_streak |
|---|---|---|---|---|---|
| above_vwap | 1162 | 0.353 | -0.011 | 0.983 | 27 |
| below_vwap | 1124 | 0.369 | 0.038 | 1.056 | 16 |

### 1H Trend Alignment

Bucket definition:
- `aligned_with_reversal`: `ema_slope_1h` searah posisi reversal
- `aligned_with_breakout`: `ema_slope_1h` searah breakout (lawan reversal)
- `flat_or_unknown`: slope 0/NA

| bucket | n | win_rate | exp_net_R | pf_net | max_loss_streak |
|---|---|---|---|---|---|
| aligned_with_reversal | 1182 | 0.387 | 0.092 | 1.141 | 18 |
| aligned_with_breakout | 1104 | 0.332 | -0.073 | 0.898 | 22 |

## Reversal vs Continuation by Year (All signals)

| bucket | n | win_rate | exp_net_R | pf_net |
|---|---|---|---|---|
| 2024_cont | 2301 | 0.329 | -0.082 | 0.886 |
| 2024_rev | 2301 | 0.349 | -0.023 | 0.967 |
| 2025_cont | 2281 | 0.362 | 0.016 | 1.024 |
| 2025_rev | 2281 | 0.316 | -0.123 | 0.832 |
| 2026_cont | 709 | 0.310 | -0.139 | 0.811 |
| 2026_rev | 709 | 0.343 | -0.042 | 0.941 |
