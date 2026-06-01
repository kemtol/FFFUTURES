# MNQ ORB ST5_50 Variant Comparison

## Scope

| Field | Value |
| --- | --- |
| ORB | 15m New York opening range |
| Exit | TP 2R or 15:00 NY EOD/time exit |
| Risk | $500 target risk |
| ST variant | ST5_50, factor 4.0 |
| Long ST rule | `ST5_50_dir == -1` bullish/up |
| Short ST rule | `ST5_50_dir == +1` bearish/down |
| Join rule | Latest completed ST timestamp `<= signal_ts` |
| Anchor | 2026-05-28T01:53:00+00:00 |
| Lookahead violations | 0 |

## Comparison

| Variant | Trades | Long | Short | WR | PnL | DD | Ret/DD | Jan-May Trades | Jan-May PnL | Jan-May DD | Mar PnL | Mar DD | 30D Trades | 30D PnL | 30D DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long only, no ST | 1296 | 1296 | 0 | 56.48% | $33,091 | $-12,124 | 2.73 | 72 | $6,096 | $-4,085 | $-2,633 | $-4,085 | 18 | $3,460 | $-551 |
| Long only, ST5_50 bullish | 964 | 964 | 0 | 56.85% | $28,199 | $-8,027 | 3.51 | 50 | $5,843 | $-1,916 | $102 | $-1,588 | 15 | $1,918 | $-551 |
| Long+Short, no ST | 1767 | 905 | 862 | 53.03% | $26,501 | $-15,294 | 1.73 | 97 | $4,072 | $-3,623 | $-1,451 | $-3,105 | 21 | $839 | $-1,636 |
| Long+Short, ST5_50 aligned | 1251 | 667 | 584 | 53.88% | $36,800 | $-9,099 | 4.04 | 63 | $7,328 | $-4,493 | $2,336 | $-1,029 | 16 | $-1,059 | $-1,894 |

## Current Read

- `ST5_50` as long-only filter is the clean P0 candidate because it fixes March with only one extra rule.
- `Long+Short no ST` adds frequency, but the short leg is weak by itself.
- `Long+Short ST5_50 aligned` is worth tracking only if it improves drawdown without diluting the long-only edge.

## Artifacts

| Artifact | Path |
| --- | --- |
| CSV | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/supertrend_variant_comparison.csv` |
| Manifest | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_variant_comparison_manifest.json` |
