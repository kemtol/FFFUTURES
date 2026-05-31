# MNQ ORB Risk-Adjusted V2 Report

Created: `2026-05-30T10:06:02.502320+00:00`

## Dataset

- Rows: `2559`
- Columns: `77`
- Features: `62`
- Train/validation/holdout: `{'holdout': 140, 'train': 2068, 'validation': 351}`
- Daily confluence lookahead violations: `0`

## Feature Families

| Family | Features |
| --- | ---: |
| `volatility_atr` | 9 |
| `vwap_context` | 4 |
| `overnight_structure` | 5 |
| `breakout_quality` | 11 |
| `prior_day_context` | 4 |
| `daily_confluence` | 29 |

## V1 vs V2 Holdout

| Target | Model | Metric | V1 | V2 | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| `success_2r` | `logistic` | `auc` | 0.716 | 0.716 | -0.001 |
| `success_2r` | `logistic` | `average_precision` | 0.350 | 0.364 | 0.014 |
| `success_2r` | `logistic` | `brier` | 0.197 | 0.212 | 0.015 |
| `success_2r` | `lgbm_shallow` | `auc` | 0.606 | 0.633 | 0.027 |
| `success_2r` | `lgbm_shallow` | `average_precision` | 0.211 | 0.236 | 0.025 |
| `success_2r` | `lgbm_shallow` | `brier` | 0.140 | 0.138 | -0.002 |
| `positive_eod` | `logistic` | `auc` | 0.542 | 0.584 | 0.042 |
| `positive_eod` | `logistic` | `average_precision` | 0.624 | 0.641 | 0.016 |
| `positive_eod` | `logistic` | `brier` | 0.249 | 0.246 | -0.003 |
| `positive_eod` | `lgbm_shallow` | `auc` | 0.550 | 0.515 | -0.034 |
| `positive_eod` | `lgbm_shallow` | `average_precision` | 0.646 | 0.588 | -0.057 |
| `positive_eod` | `lgbm_shallow` | `brier` | 0.242 | 0.245 | 0.003 |

## Models

### Target `success_2r`

| Model | Val AUC | Val PR-AUC | Val Brier | Holdout AUC | Holdout PR-AUC | Holdout Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logistic` | 0.739 | 0.375 | 0.223 | 0.716 | 0.364 | 0.212 |
| `lgbm_shallow` | 0.690 | 0.265 | 0.126 | 0.633 | 0.236 | 0.138 |

### Target `positive_eod`

| Model | Val AUC | Val PR-AUC | Val Brier | Holdout AUC | Holdout PR-AUC | Holdout Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logistic` | 0.551 | 0.618 | 0.250 | 0.584 | 0.641 | 0.246 |
| `lgbm_shallow` | 0.576 | 0.604 | 0.242 | 0.515 | 0.588 | 0.245 |

## Top Feature Families

### `success_2r`

#### `logistic`

| Family | Importance Sum | Importance Mean |
| --- | ---: | ---: |
| `daily_confluence` | 4.8655 | 0.1678 |
| `volatility_atr` | 2.5525 | 0.2836 |
| `overnight_structure` | 1.9393 | 0.3879 |
| `breakout_quality` | 1.4736 | 0.1340 |
| `vwap_context` | 0.3510 | 0.0878 |
| `prior_day_context` | 0.2760 | 0.0690 |

#### `lgbm_shallow`

| Family | Importance Sum | Importance Mean |
| --- | ---: | ---: |
| `volatility_atr` | 547.8728 | 60.8748 |
| `breakout_quality` | 309.9393 | 28.1763 |
| `vwap_context` | 157.1718 | 39.2929 |
| `daily_confluence` | 88.9832 | 3.0684 |
| `prior_day_context` | 0.0000 | 0.0000 |
| `overnight_structure` | 0.0000 | 0.0000 |

### `positive_eod`

#### `logistic`

| Family | Importance Sum | Importance Mean |
| --- | ---: | ---: |
| `daily_confluence` | 1.9566 | 0.0675 |
| `volatility_atr` | 0.8452 | 0.0939 |
| `prior_day_context` | 0.5726 | 0.1432 |
| `breakout_quality` | 0.4613 | 0.0419 |
| `overnight_structure` | 0.4177 | 0.0835 |
| `vwap_context` | 0.0633 | 0.0158 |

#### `lgbm_shallow`

| Family | Importance Sum | Importance Mean |
| --- | ---: | ---: |
| `daily_confluence` | 248.5093 | 8.5693 |
| `breakout_quality` | 144.5252 | 13.1387 |
| `vwap_context` | 118.5544 | 29.6386 |
| `volatility_atr` | 42.7384 | 4.7487 |
| `prior_day_context` | 23.7973 | 5.9493 |
| `overnight_structure` | 13.0334 | 2.6067 |

## Daily Confluence Readout

Top daily confluence features from `success_2r` logistic by absolute coefficient:

| Feature | Importance | Signed Value |
| --- | ---: | ---: |
| `dc_spy_realized_vol_20d` | 0.6128 | -0.6128 |
| `dc_vix_prev_close` | 0.4948 | 0.4948 |
| `dc_spy_dist_sma20` | 0.4795 | -0.4795 |
| `dc_qqq_dist_sma50` | 0.3774 | -0.3774 |
| `dc_qqq_dist_sma20` | 0.3300 | 0.3300 |
| `dc_spy_dist_sma50` | 0.3114 | 0.3114 |
| `dc_vix_change_1d` | 0.2870 | -0.2870 |
| `dc_tnx_prev_close` | 0.2632 | 0.2632 |
| `dc_qqq_beta_residual_20d` | 0.2449 | 0.2449 |
| `dc_qqq_minus_spy_return_20d` | 0.2302 | -0.2302 |

## Holdout Slices

Primary diagnostic shown for `success_2r` logistic.

### Side

| Side | Rows | Target Rate | Avg Prob | AUC | Brier | Sum PnL/ct | DD/ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DOWN` | 65 | 0.200 | 0.517 | 0.604 | 0.281 | $2,452 | $-2,894 |
| `UP` | 75 | 0.147 | 0.359 | 0.835 | 0.152 | $4,153 | $-2,530 |

### `vix_level`

| Tertile | Rows | Target Rate | Avg Prob | AUC | Brier | Sum PnL/ct | DD/ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `high` | 34 | 0.206 | 0.580 | 0.672 | 0.300 | $2,012 | $-1,371 |
| `low` | 22 | 0.136 | 0.341 | 0.702 | 0.176 | $1,428 | $-880 |
| `mid` | 84 | 0.167 | 0.397 | 0.753 | 0.187 | $3,164 | $-2,006 |

### `qqq_relative_strength_5d`

| Tertile | Rows | Target Rate | Avg Prob | AUC | Brier | Sum PnL/ct | DD/ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `high` | 59 | 0.169 | 0.470 | 0.659 | 0.246 | $1,368 | $-1,394 |
| `low` | 28 | 0.179 | 0.396 | 0.713 | 0.193 | $3,424 | $-581 |
| `mid` | 53 | 0.170 | 0.410 | 0.742 | 0.185 | $1,812 | $-1,078 |

### `dxy_trend_5d`

| Tertile | Rows | Target Rate | Avg Prob | AUC | Brier | Sum PnL/ct | DD/ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `high` | 55 | 0.145 | 0.417 | 0.673 | 0.220 | $4,026 | $-888 |
| `low` | 42 | 0.190 | 0.485 | 0.768 | 0.224 | $1,690 | $-1,403 |
| `mid` | 43 | 0.186 | 0.400 | 0.761 | 0.191 | $889 | $-1,171 |

## Readout

- V2 uses the same model families as V1, but on the 62-feature confluence dataset.
- V1 metrics are retained only as a pre-confluence baseline.
- This is still research. Kelly and Topstep-style overlays must be evaluated before live forward testing.
