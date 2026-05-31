# MNQ ORB 15m Long TP2R/EOD Rule-Based

Strategy ID: `rule_based_15m_long_tp2r_eod`

## Contract

| Field | Value |
| --- | --- |
| Instrument | MNQ |
| Session | New York regular session |
| Source grain | Right-labeled M1 bars |
| Opening range | 15 minutes after 09:30 NY |
| Direction | Long only |
| Signal | First M1 close above OR high |
| Entry | Next M1 open after signal close |
| Exit | TP 2R first, otherwise 15:00 NY EOD/time exit |
| Strategic SL | None |
| Stop reference | OR low for position sizing only |
| Max trades | 1 per NY session |

## Cost Model

| Cost | Value |
| --- | ---: |
| Commission + fees | $1.24 RT / contract |
| Slippage | 1 tick per side |
| Modeled slippage | $1.00 RT / contract |
| Total commission paid | $5,590 |
| Total modeled slippage | $4,508 |

## Performance

| Metric | Value |
| --- | ---: |
| Signal range | 2019-05-06 to 2026-05-26 |
| Trades | 1,296 |
| Win rate | 56.48% |
| Total PnL | $33,091 |
| Gross profit | $314,667 |
| Gross loss | $-281,576 |
| Profit factor | 1.12 |
| Max drawdown | $-12,124 |
| Return / DD | 2.73 |
| Expectancy / trade | $25.53 |
| Median trade | $93.01 |
| Average win | $429.87 |
| Average loss | $-499.25 |
| Payoff ratio | 0.86 |
| Average contracts | 3.48 |
| Max consecutive wins | 10 |
| Max consecutive losses | 6 |

## Daily Quality

Sharpe and Sortino are computed from daily dollar PnL over MNQ NY session days, with zero PnL on no-trade days, annualized by `sqrt(252)`.

| Metric | Value |
| --- | ---: |
| Trading days measured | 2,197 |
| Active days | 1,296 |
| Active-day rate | 58.99% |
| Active-day win rate | 56.48% |
| Daily average PnL | $15.06 |
| Daily PnL std dev | $479.24 |
| Daily Sharpe | 0.50 |
| Daily Sortino | 0.64 |
| Best day | $991 |
| Worst day | $-3,781 |
| Best-day profit share | 2.99% |
| 50% consistency flag | Pass |

## Rolling Windows

| Window | Trades | Win Rate | PnL | Max DD |
| ---: | ---: | ---: | ---: | ---: |
| 5D | 1 | 100.00% | $101 | $0 |
| 10D | 5 | 60.00% | $895 | $-222 |
| 20D | 10 | 70.00% | $2,348 | $-222 |
| 30D | 18 | 72.22% | $3,460 | $-551 |
| 50D | 30 | 66.67% | $5,385 | $-861 |
| 100D | 54 | 55.56% | $4,035 | $-4,085 |
| 200D | 94 | 61.70% | $5,385 | $-4,561 |

## Readout

- This is the dedicated artifact for the current cleanest MNQ ORB rule-based edge.
- The strategy intentionally has no normal SL: it exits at TP 2R or 15:00 NY.
- OR low is used only to size contracts; it is not a simulated stop exit.
- Catastrophic safety guard results are in `flash_guard_report.md`.
- Live promotion still needs Topstep MLL, first-$3000 path, and forward-test checks.
