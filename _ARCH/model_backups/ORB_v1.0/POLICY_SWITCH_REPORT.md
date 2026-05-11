# Policy Switch Evaluation — ORB_v1.0

Generated: 2026-05-06 11:45

## Policy

- Models: `lgbm_rev_1r2_120m` + `lgbm_cont_1r2_120m`
- Holdout: `2024-01-01` onward
- Calibration window: `2022-01-01` to `2023-12-31`
- Thresholds:
  - `t_rev` (p60): `0.328765`
  - `t_cont` (p60): `0.317631`
  - `q75_rev`: `0.527349`
  - `q75_cont`: `0.496089`
- Decision logic:
  - `rev` candidate if `prob_rev >= t_rev`, trend aligns with reversal, and (`ADX>=30` or `prob_rev>=q75_rev`)
  - `cont` candidate if `prob_cont >= t_cont`, trend aligns with breakout, and (`ADX<50` or `prob_cont>=q75_cont`)
  - if both candidates valid: pick higher probability
  - if none valid: `skip`

## Overall (Holdout 2024+)

| policy | n | win_rate | exp_net_R | pf_net | sharpe_trade_net | max_loss_streak |
|---|---|---|---|---|---|---|
| always_rev | 5291 | 0.334 | -0.069 | 0.904 | -3.531 | 33 |
| always_cont | 5291 | 0.341 | -0.047 | 0.933 | -2.410 | 26 |
| max_prob_no_gate | 5291 | 0.359 | 0.007 | 1.010 | 0.340 | 24 |
| dynamic_rev_cont_skip | 2226 | 0.397 | 0.121 | 1.188 | 3.900 | 17 |

## Decision Mix (Dynamic Policy)

| decision | share |
|---|---|
| skip | 0.579 |
| cont | 0.298 |
| rev | 0.122 |

## Dynamic Policy by Year

| policy | n | win_rate | exp_net_R | pf_net | max_loss_streak |
|---|---|---|---|---|---|
| 2024 | 1096 | 0.411 | 0.164 | 1.261 | 11 |
| 2025 | 961 | 0.396 | 0.119 | 1.185 | 14 |
| 2026 | 169 | 0.308 | -0.147 | 0.802 | 17 |

## Dynamic Policy by Regime

| policy | n | win_rate | exp_net_R | pf_net | max_loss_streak |
|---|---|---|---|---|---|
| adx_20-30 | 659 | 0.420 | 0.191 | 1.308 | 14 |
| adx_30-50 | 862 | 0.378 | 0.065 | 1.097 | 14 |
| adx_<20 | 503 | 0.400 | 0.129 | 1.201 | 14 |
| adx_>50 | 202 | 0.396 | 0.118 | 1.183 | 14 |
| trend_breakout | 1578 | 0.395 | 0.116 | 1.180 | 17 |
| trend_reversal | 648 | 0.401 | 0.134 | 1.209 | 19 |
| above_vwap | 1218 | 0.398 | 0.125 | 1.193 | 16 |
| below_vwap | 1008 | 0.396 | 0.117 | 1.182 | 19 |

## North Star Check (Dynamic Policy)

- PF net > 1.3: `False`
- Sharpe trade net > 1.0: `True`
- Max loss streak < 8: `False`
