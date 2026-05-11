# Super Structure Parity 2026-05-07 (Asia/Jakarta)

Window UTC: `2026-05-06T17:00:00+00:00` -> `2026-05-07T17:00:00+00:00`

UI rows are theoretical backtest rows from the TopstepX buffer snapshot; they are not execution truth.

## Signal vs Topstep
| severity | signal | signal_ts | signal_px | execution | exec_ts | exec_px | slippage | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | BUY | 2026-05-07 01:25 | 4721.20 | MARKET | 2026-05-07 01:25 | 4720.00 | -1.20 |  |
| PASS | CLOSE | 2026-05-07 04:40 | 4702.64 | FLATTEN | 2026-05-07 04:40 | 4702.64 | 0.00 |  |
| PASS | SELL | 2026-05-07 04:40 | 4701.50 | MARKET | 2026-05-07 04:40 | 4701.00 | -0.50 |  |
| PASS | CLOSE | 2026-05-07 06:25 | 4717.44 | FLATTEN | 2026-05-07 06:25 | 4717.44 | 0.00 |  |
| PASS | BUY | 2026-05-07 06:35 | 4726.70 | MARKET | 2026-05-07 06:35 | 4726.50 | -0.20 |  |
| PASS | SELL | 2026-05-07 15:55 | 4742.10 | MARKET | 2026-05-07 15:55 | 4740.70 | -1.40 |  |

## Signal vs UI Theoretical
| severity | side | signal_entry | ui_entry | entry_delta_min | exit_delta_min | entry_px_delta | exit_px_delta | pnl_delta | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | 2026-05-07 01:25 | 2026-05-07 01:55 | -30.00 | 5.00 | -4.80 | -0.00 | 48.00 | UI is theoretical backtest |
| PASS | Short | 2026-05-07 04:40 | 2026-05-07 04:35 | 5.00 | 10.00 | -0.70 | 0.00 | -7.00 | UI is theoretical backtest |
| PASS | Long | 2026-05-07 06:35 | 2026-05-07 06:30 | 5.00 |  | 0.50 |  |  | Live was manually closed; signal archive has no CLOSE |
| CRITICAL | Short | 2026-05-07 15:55 | MISSING |  |  |  |  |  | no UI theoretical trade within 30min |

## Topstep vs UI Theoretical
| severity | side | actual_entry | ui_entry | entry_delta_min | exit_delta_min | entry_px_delta | exit_px_delta | pnl_delta | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | 2026-05-07 01:25 | 2026-05-07 01:55 | -30.00 | 5.00 | -6.00 | -0.00 | 60.00 | UI is theoretical backtest |
| PASS | Short | 2026-05-07 04:40 | 2026-05-07 04:35 | 5.00 | 10.00 | -1.20 | 0.00 | -12.00 | UI is theoretical backtest |
| PASS | Long | 2026-05-07 06:35 | 2026-05-07 06:30 | 5.00 |  | 0.30 |  |  | actual was manually closed; exact close time unavailable in legacy log |
| CRITICAL | Short | 2026-05-07 15:55 | MISSING |  |  |  |  |  | actual execution without matching UI theoretical trade |
