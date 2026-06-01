# NASDAQ ORB Short Breakout Switch-To-Long Comparison

## Scope

| Field | Value |
| --- | --- |
| Instrument | NASDAQ Micro Futures (`MNQ`) |
| ORB | 15m New York opening range |
| Long rule | Baseline long continuation, TP 2R or 15:00 NY EOD |
| Short rule | If OR low breaks first, enter short; if price closes above OR high, close short and switch to long next M1 open |
| Short TP sweep | 1R, 1.5R, 2R |
| Cost model | TopstepX MNQ commission + 1 tick slippage per side |
| Lookahead handling | Reversal is based on M1 close; switch execution uses next M1 open |
| Anchor | 2026-05-28T01:53:00+00:00 |

## Charts

### Equity Curve

![Short Switch Equity](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_reversal_switch_equity_curve.png)

### Drawdown Curve

![Short Switch Drawdown](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_reversal_switch_drawdown_curve.png)

### Last 30D Equity

![Short Switch Last 30D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_reversal_switch_last30_equity.png)

## Summary

| Variant | Trades | WR | PnL | DD | Ret/DD | Short-first | Switches | Short PnL | Jan-May PnL | Mar PnL | 30D PnL | 30D DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long only, no ST | 1,296 | 56.48% | $33,091 | $-12,124 | 2.73 | 0 | 0 | $0 | $6,096 | $-2,633 | $3,460 | $-551 |
| Long+Short first breakout, no switch | 1,767 | 53.03% | $26,501 | $-15,294 | 1.73 | 862 | 0 | $0 | $4,072 | $-1,451 | $839 | $-1,636 |
| Short switch to long, short TP 1.5R | 1,767 | 54.90% | $26,124 | $-16,047 | 1.63 | 862 | 375 | $-7,598 | $6,592 | $-2,085 | $1,323 | $-886 |
| Short switch to long, short TP 1R | 1,767 | 57.89% | $32,242 | $-13,722 | 2.35 | 862 | 340 | $-607 | $4,270 | $-2,480 | $973 | $-886 |
| Short switch to long, short TP 2R | 1,767 | 53.99% | $37,731 | $-12,715 | 2.97 | 862 | 382 | $3,671 | $7,085 | $-2,074 | $1,515 | $-886 |

## Last 30D Detail - Switch Variants

### Short TP 1R

| NY Date | First Side | Switched | Legs | Exit Path | PnL | Short PnL | Long PnL |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 2026-04-28 | LONG | No | 1 | TIME_EXIT | $-107 | $0 | $-107 |
| 2026-04-29 | LONG | No | 1 | TIME_EXIT | $276 | $0 | $276 |
| 2026-04-30 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-360 | $-470 | $109 |
| 2026-05-01 | LONG | No | 1 | TIME_EXIT | $26 | $0 | $26 |
| 2026-05-04 | LONG | No | 1 | TIME_EXIT | $-551 | $0 | $-551 |
| 2026-05-05 | LONG | No | 1 | TIME_EXIT | $289 | $0 | $289 |
| 2026-05-06 | LONG | No | 1 | TIME_EXIT | $288 | $0 | $288 |
| 2026-05-07 | SHORT | Yes | 2 | SWITCH_TO_LONG+TP_2R | $326 | $-458 | $784 |
| 2026-05-08 | LONG | No | 1 | TIME_EXIT | $397 | $0 | $397 |
| 2026-05-11 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-460 | $-358 | $-102 |
| 2026-05-12 | SHORT | No | 1 | TP_1R | $302 | $302 | $0 |
| 2026-05-13 | SHORT | Yes | 2 | SWITCH_TO_LONG+TP_2R | $390 | $-494 | $885 |
| 2026-05-14 | LONG | No | 1 | TIME_EXIT | $98 | $0 | $98 |
| 2026-05-15 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-162 | $-337 | $175 |
| 2026-05-18 | SHORT | No | 1 | TP_1R | $394 | $394 | $0 |
| 2026-05-19 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-692 | $-470 | $-222 |
| 2026-05-20 | LONG | No | 1 | TIME_EXIT | $213 | $0 | $213 |
| 2026-05-21 | LONG | No | 1 | TP_2R | $979 | $0 | $979 |
| 2026-05-22 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-603 | $-429 | $-175 |
| 2026-05-26 | LONG | No | 1 | TIME_EXIT | $101 | $0 | $101 |
| 2026-05-27 | SHORT | No | 1 | TIME_EXIT | $-166 | $-166 | $0 |

### Short TP 1.5R

| NY Date | First Side | Switched | Legs | Exit Path | PnL | Short PnL | Long PnL |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 2026-04-28 | LONG | No | 1 | TIME_EXIT | $-107 | $0 | $-107 |
| 2026-04-29 | LONG | No | 1 | TIME_EXIT | $276 | $0 | $276 |
| 2026-04-30 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-360 | $-470 | $109 |
| 2026-05-01 | LONG | No | 1 | TIME_EXIT | $26 | $0 | $26 |
| 2026-05-04 | LONG | No | 1 | TIME_EXIT | $-551 | $0 | $-551 |
| 2026-05-05 | LONG | No | 1 | TIME_EXIT | $289 | $0 | $289 |
| 2026-05-06 | LONG | No | 1 | TIME_EXIT | $288 | $0 | $288 |
| 2026-05-07 | SHORT | Yes | 2 | SWITCH_TO_LONG+TP_2R | $326 | $-458 | $784 |
| 2026-05-08 | LONG | No | 1 | TIME_EXIT | $397 | $0 | $397 |
| 2026-05-11 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-460 | $-358 | $-102 |
| 2026-05-12 | SHORT | No | 1 | TP_1.5R | $454 | $454 | $0 |
| 2026-05-13 | SHORT | Yes | 2 | SWITCH_TO_LONG+TP_2R | $390 | $-494 | $885 |
| 2026-05-14 | LONG | No | 1 | TIME_EXIT | $98 | $0 | $98 |
| 2026-05-15 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-162 | $-337 | $175 |
| 2026-05-18 | SHORT | No | 1 | TP_1.5R | $592 | $592 | $0 |
| 2026-05-19 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-692 | $-470 | $-222 |
| 2026-05-20 | LONG | No | 1 | TIME_EXIT | $213 | $0 | $213 |
| 2026-05-21 | LONG | No | 1 | TP_2R | $979 | $0 | $979 |
| 2026-05-22 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-603 | $-429 | $-175 |
| 2026-05-26 | LONG | No | 1 | TIME_EXIT | $101 | $0 | $101 |
| 2026-05-27 | SHORT | No | 1 | TIME_EXIT | $-166 | $-166 | $0 |

### Short TP 2R

| NY Date | First Side | Switched | Legs | Exit Path | PnL | Short PnL | Long PnL |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 2026-04-28 | LONG | No | 1 | TIME_EXIT | $-107 | $0 | $-107 |
| 2026-04-29 | LONG | No | 1 | TIME_EXIT | $276 | $0 | $276 |
| 2026-04-30 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-360 | $-470 | $109 |
| 2026-05-01 | LONG | No | 1 | TIME_EXIT | $26 | $0 | $26 |
| 2026-05-04 | LONG | No | 1 | TIME_EXIT | $-551 | $0 | $-551 |
| 2026-05-05 | LONG | No | 1 | TIME_EXIT | $289 | $0 | $289 |
| 2026-05-06 | LONG | No | 1 | TIME_EXIT | $288 | $0 | $288 |
| 2026-05-07 | SHORT | Yes | 2 | SWITCH_TO_LONG+TP_2R | $326 | $-458 | $784 |
| 2026-05-08 | LONG | No | 1 | TIME_EXIT | $397 | $0 | $397 |
| 2026-05-11 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-460 | $-358 | $-102 |
| 2026-05-12 | SHORT | No | 1 | TP_2R | $606 | $606 | $0 |
| 2026-05-13 | SHORT | Yes | 2 | SWITCH_TO_LONG+TP_2R | $390 | $-494 | $885 |
| 2026-05-14 | LONG | No | 1 | TIME_EXIT | $98 | $0 | $98 |
| 2026-05-15 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-162 | $-337 | $175 |
| 2026-05-18 | SHORT | No | 1 | TIME_EXIT | $632 | $632 | $0 |
| 2026-05-19 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-692 | $-470 | $-222 |
| 2026-05-20 | LONG | No | 1 | TIME_EXIT | $213 | $0 | $213 |
| 2026-05-21 | LONG | No | 1 | TP_2R | $979 | $0 | $979 |
| 2026-05-22 | SHORT | Yes | 2 | SWITCH_TO_LONG+TIME_EXIT | $-603 | $-429 | $-175 |
| 2026-05-26 | LONG | No | 1 | TIME_EXIT | $101 | $0 | $101 |
| 2026-05-27 | SHORT | No | 1 | TIME_EXIT | $-166 | $-166 | $0 |

## Current Read

- This test matches the intended asymmetric NASDAQ logic: short is allowed, but a failed short must yield to long continuation.
- The key comparison is not whether short can trade more often, but whether the short leg improves drawdown without stealing the natural long bias.
- If all short TP variants still degrade 30D or drawdown, short should remain a separate exploratory module rather than part of the primary ORB.

## Artifacts

| Artifact | Path |
| --- | --- |
| Summary CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_reversal_switch_comparison.csv` |
| Sequence events CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_reversal_switch_events.csv` |
| Leg-level CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_reversal_switch_legs.csv` |
| Manifest | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_reversal_switch_comparison_manifest.json` |
