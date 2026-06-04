# NASDAQ ORB Short-Switch TP2R P0 Sweep

## Scope

| Field | Value |
| --- | --- |
| Strategy family | NASDAQ Micro Futures ORB 15m short-switch TP2R |
| Long-first risk | $500 fixed |
| Short TP | 2R |
| Long TP after switch | 2R or 15:00 NY |
| Short filters | `none`, `st5_50_bearish`, `st5_20_bearish`, `st5_50_and_st15_20_bearish` |
| Short risk grid | $250, $350, $500 |
| Switch long risk grid | $500, $750 |
| Switch buffers | `0`, `2ticks`, `0.25r` |
| Short time guards | `none`, `10:30`, `11:00` |
| No-lookahead rule | ST feature timestamp is selected by as-of `<= signal_ts`; entry/switch executes next M1 open |
| Variants evaluated | 216 plus baseline |

## Baseline Comparison

| Variant | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long only, no ST | 1,296 | $33,091 | $-12,124 | 2.73 | $-2,633 | 18 | $3,460 | $-551 |
| Existing short-switch TP2R equivalent | 1,767 | $37,731 | $-12,715 | 2.97 | $-2,074 | 21 | $1,515 | $-886 |
| Best P0 score | 1,660 | $42,946 | $-12,792 | 3.36 | $-975 | 16 | $4,696 | $-551 |

## Best P0 Candidate

| Field | Value |
| --- | --- |
| Variant ID | `p0_st5_20_bearish_sr350_lr750_buf0_tg1030` |
| Short filter | `st5_20_bearish` |
| Short risk | $350 |
| Switch long risk | $750 |
| Switch buffer | `0` |
| Short entry guard | `10:30` |
| 30D PnL delta vs baseline | $1,236 |
| 30D DD delta vs baseline | $0 |
| March PnL delta vs baseline | $1,658 |
| Beats baseline 30D PnL | True |
| Beats baseline 30D DD | False |
| Improves March PnL | True |

## Top By Recent 30D PnL

| Variant | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD | Short PnL | Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p0_st5_50_bearish_sr350_lr750_buf0_tgnone | 1,711 | $38,950 | $-14,135 | 2.76 | $-975 | 15 | $4,724 | $-551 | $4,157 | 306 |
| p0_st5_50_bearish_sr350_lr750_buf0_tg1100 | 1,694 | $41,679 | $-13,055 | 3.19 | $-975 | 15 | $4,724 | $-551 | $6,404 | 300 |
| p0_st5_50_bearish_sr350_lr750_buf2ticks_tgnone | 1,711 | $35,744 | $-15,753 | 2.27 | $-975 | 15 | $4,724 | $-551 | $3,408 | 305 |
| p0_st5_50_bearish_sr350_lr750_buf2ticks_tg1100 | 1,694 | $38,474 | $-14,370 | 2.68 | $-975 | 15 | $4,724 | $-551 | $5,655 | 299 |
| p0_none_sr350_lr750_buf0_tgnone | 1,723 | $37,333 | $-14,871 | 2.51 | $-2,555 | 15 | $4,706 | $-551 | $1,848 | 363 |
| p0_none_sr350_lr750_buf0_tg1100 | 1,712 | $39,318 | $-13,661 | 2.88 | $-2,555 | 15 | $4,706 | $-551 | $4,001 | 358 |
| p0_none_sr350_lr750_buf2ticks_tgnone | 1,723 | $34,032 | $-17,373 | 1.96 | $-2,555 | 15 | $4,706 | $-551 | $1,031 | 362 |
| p0_none_sr350_lr750_buf2ticks_tg1100 | 1,712 | $36,017 | $-16,167 | 2.23 | $-2,555 | 15 | $4,706 | $-551 | $3,183 | 357 |
| p0_st5_20_bearish_sr350_lr750_buf0_tgnone | 1,713 | $40,368 | $-14,158 | 2.85 | $-975 | 16 | $4,696 | $-551 | $5,564 | 284 |
| p0_st5_20_bearish_sr350_lr750_buf0_tg1030 | 1,660 | $42,946 | $-12,792 | 3.36 | $-975 | 16 | $4,696 | $-551 | $8,678 | 272 |
| p0_st5_20_bearish_sr350_lr750_buf0_tg1100 | 1,692 | $42,884 | $-13,139 | 3.26 | $-975 | 16 | $4,696 | $-551 | $8,127 | 277 |
| p0_st5_20_bearish_sr350_lr750_buf2ticks_tgnone | 1,713 | $37,159 | $-15,390 | 2.41 | $-975 | 16 | $4,696 | $-551 | $4,887 | 283 |

## Top By Full Ret/DD

| Variant | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD | Short PnL | Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p0_st5_20_bearish_sr500_lr500_buf0_tg1030 | 1,706 | $50,558 | $-10,410 | 4.86 | $-1,366 | 20 | $2,649 | $-886 | $15,272 | 292 |
| p0_st5_20_bearish_sr500_lr500_buf0_tg1100 | 1,739 | $49,759 | $-10,539 | 4.72 | $-1,366 | 20 | $2,649 | $-886 | $14,472 | 297 |
| p0_st5_50_bearish_sr500_lr500_buf0_tg1100 | 1,742 | $47,738 | $-10,474 | 4.56 | $-1,366 | 20 | $2,226 | $-886 | $12,610 | 321 |
| p0_st5_50_bearish_sr500_lr500_buf0_tg1030 | 1,712 | $47,226 | $-10,606 | 4.45 | $-1,366 | 20 | $2,584 | $-886 | $12,098 | 314 |
| p0_st5_50_and_st15_20_bearish_sr500_lr500_buf0_tg1030 | 1,640 | $44,857 | $-10,190 | 4.40 | $-2,135 | 19 | $2,809 | $-916 | $11,573 | 207 |
| p0_st5_50_and_st15_20_bearish_sr500_lr500_buf0_tg1100 | 1,684 | $44,154 | $-10,314 | 4.28 | $-2,135 | 19 | $2,809 | $-916 | $10,869 | 214 |
| p0_st5_50_and_st15_20_bearish_sr500_lr500_buf2ticks_tg1030 | 1,640 | $43,792 | $-10,583 | 4.14 | $-2,135 | 19 | $2,809 | $-916 | $11,026 | 206 |
| p0_st5_50_and_st15_20_bearish_sr500_lr500_buf2ticks_tg1100 | 1,684 | $43,057 | $-10,707 | 4.02 | $-2,135 | 19 | $2,809 | $-916 | $10,314 | 213 |
| p0_st5_20_bearish_sr350_lr500_buf0_tg1030 | 1,660 | $42,785 | $-10,748 | 3.98 | $-975 | 16 | $3,556 | $-551 | $8,678 | 272 |
| p0_st5_50_and_st15_20_bearish_sr500_lr500_buf0_tgnone | 1,727 | $40,435 | $-10,200 | 3.96 | $-2,135 | 19 | $2,809 | $-916 | $7,150 | 221 |
| p0_st5_20_bearish_sr500_lr500_buf2ticks_tg1030 | 1,706 | $47,894 | $-12,210 | 3.92 | $-1,366 | 20 | $2,649 | $-886 | $14,233 | 291 |
| p0_st5_20_bearish_sr500_lr750_buf0_tg1030 | 1,706 | $51,410 | $-13,134 | 3.91 | $-1,272 | 20 | $3,789 | $-886 | $15,272 | 292 |

## Top By P0 Score

| Variant | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD | Short PnL | Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p0_st5_20_bearish_sr350_lr750_buf0_tg1030 | 1,660 | $42,946 | $-12,792 | 3.36 | $-975 | 16 | $4,696 | $-551 | $8,678 | 272 |
| p0_st5_20_bearish_sr350_lr750_buf0_tg1100 | 1,692 | $42,884 | $-13,139 | 3.26 | $-975 | 16 | $4,696 | $-551 | $8,127 | 277 |
| p0_st5_50_bearish_sr350_lr750_buf0_tg1100 | 1,694 | $41,679 | $-13,055 | 3.19 | $-975 | 15 | $4,724 | $-551 | $6,404 | 300 |
| p0_st5_20_bearish_sr350_lr750_buf0_tgnone | 1,713 | $40,368 | $-14,158 | 2.85 | $-975 | 16 | $4,696 | $-551 | $5,564 | 284 |
| p0_st5_50_bearish_sr350_lr750_buf0_tg1030 | 1,666 | $40,951 | $-12,889 | 3.18 | $-975 | 16 | $4,622 | $-551 | $6,419 | 294 |
| p0_st5_20_bearish_sr350_lr750_buf2ticks_tg1030 | 1,660 | $39,762 | $-14,025 | 2.84 | $-975 | 16 | $4,696 | $-551 | $8,009 | 271 |
| p0_st5_20_bearish_sr350_lr750_buf2ticks_tg1100 | 1,692 | $39,674 | $-14,372 | 2.76 | $-975 | 16 | $4,696 | $-551 | $7,450 | 276 |
| p0_st5_50_bearish_sr350_lr750_buf0_tgnone | 1,711 | $38,950 | $-14,135 | 2.76 | $-975 | 15 | $4,724 | $-551 | $4,157 | 306 |
| p0_st5_50_bearish_sr350_lr750_buf2ticks_tg1100 | 1,694 | $38,474 | $-14,370 | 2.68 | $-975 | 15 | $4,724 | $-551 | $5,655 | 299 |
| p0_st5_20_bearish_sr500_lr750_buf0_tg1030 | 1,706 | $51,410 | $-13,134 | 3.91 | $-1,272 | 20 | $3,789 | $-886 | $15,272 | 292 |
| p0_st5_20_bearish_sr500_lr750_buf0_tg1100 | 1,739 | $51,099 | $-13,645 | 3.74 | $-1,272 | 20 | $3,789 | $-886 | $14,472 | 297 |
| p0_st5_20_bearish_sr350_lr750_buf2ticks_tgnone | 1,713 | $37,159 | $-15,390 | 2.41 | $-975 | 16 | $4,696 | $-551 | $4,887 | 283 |

## Audit

| Check | Value |
| --- | ---: |
| Total lookahead violations | 0 |
| Max short-filter feature lag minutes | 14.00 |

## Artifacts

| Artifact | Path |
| --- | --- |
| Summary CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_sweep.csv` |
| Summary parquet | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_sweep.parquet` |
| Best events CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_best_events.csv` |
| Best legs CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_best_legs.csv` |
| Best yearly CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_best_yearly.csv` |
| Best monthly CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_best_monthly.csv` |
| Full report | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_full_report.md` |
| Manifest | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_sweep_manifest.json` |

## Current Read

- P0 is considered better only if it improves the 30D/Topstep window while not worsening drawdown materially.
- Full-history PnL alone is not enough because the current objective is evaluation-window behavior.
- This is still research-only and does not change the live pipeline.
