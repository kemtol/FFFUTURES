# MNQ ORB Flash Guard Sweep

This is a catastrophic safety guard simulation, not a normal strategy SL.
The base strategy remains TP 2R or 15:00 NY time exit.

| Guard | Guard Hits | PnL | Max DD | PF | Sharpe | Sortino | 30D PnL/DD | 50D PnL/DD | 100D PnL/DD | 200D PnL/DD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 0 | $33,091 | $-12,124 | 1.12 | 0.50 | 0.64 | $3,460 / $-551 | $5,385 / $-861 | $4,035 / $-4,085 | $5,385 / $-4,561 |
| $1,000 | 150 | $36,864 | $-11,849 | 1.14 | 0.62 | 0.87 | $3,008 / $-1,003 | $4,791 / $-1,244 | $3,615 / $-3,911 | $7,853 / $-3,911 |
| $1,500 | 72 | $27,291 | $-10,780 | 1.10 | 0.43 | 0.56 | $3,460 / $-551 | $4,743 / $-1,503 | $3,393 / $-4,085 | $6,631 / $-4,085 |
| $2,000 | 34 | $26,299 | $-10,366 | 1.09 | 0.40 | 0.52 | $3,460 / $-551 | $5,385 / $-861 | $4,035 / $-4,085 | $6,841 / $-4,085 |
| $3,000 | 10 | $31,316 | $-12,785 | 1.11 | 0.47 | 0.60 | $3,460 / $-551 | $5,385 / $-861 | $4,035 / $-4,085 | $5,841 / $-4,105 |

Readout: use this table to pick a live safety threshold only after Topstep MLL and forward-test checks.
