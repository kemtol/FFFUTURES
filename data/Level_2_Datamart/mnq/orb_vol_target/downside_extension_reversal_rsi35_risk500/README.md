# MNQ ORB Downside Extension Long Reversal Sweep

Created: `2026-05-30T23:29:36.903491+00:00`

## Contract

- Long-only reversal after downside extension below the 15m opening range.
- Entry modes: `touch_next_open` and `close_next_open`.
- TP is whichever is hit first: dynamic session VWAP or fixed +2R from entry.
- SL is one OR range below entry for the default sweep.
- Same-bar SL/TP ambiguity is resolved conservatively: SL first.
- RSI filter: `RSI14 <= 35`.

## Top Rows

| Rank | Ext R | Entry | RSI Max | Risk | Trades | Win Rate | PnL | Max DD | R/DD | 5D | 10D | 20D | 50D | 100D | 200D |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | `close_next_open` | 35 | $500 | 282 | 39.7% | $-11,932 | $-16,848 | -0.71 | $0 / $0 | $0 / $0 | $212 / $0 | $2,956 / $0 | $4,070 / $-226 | $2,912 / $-1,546 |
| 2 | 2 | `touch_next_open` | 35 | $500 | 303 | 43.6% | $-4,964 | $-9,369 | -0.53 | $0 / $0 | $0 / $0 | $187 / $0 | $2,419 / $0 | $3,005 / $-470 | $1,848 / $-1,628 |
| 3 | 1.5 | `touch_next_open` | 35 | $500 | 428 | 42.8% | $-10,741 | $-16,250 | -0.66 | $0 / $0 | $-72 / $-72 | $-356 / $-356 | $128 / $-842 | $719 / $-842 | $820 / $-1,501 |
| 4 | 1.5 | `close_next_open` | 35 | $500 | 402 | 41.0% | $-12,952 | $-15,243 | -0.85 | $0 / $0 | $-15 / $-15 | $-300 / $-300 | $185 / $-786 | $814 / $-786 | $255 / $-1,412 |
| 5 | 1 | `touch_next_open` | 35 | $500 | 603 | 46.6% | $-17,937 | $-19,438 | -0.92 | $0 / $0 | $-236 / $-236 | $-236 / $-236 | $-464 / $-1,616 | $-1,674 / $-1,674 | $-2,885 / $-3,966 |
| 6 | 1 | `close_next_open` | 35 | $500 | 593 | 44.7% | $-17,502 | $-18,824 | -0.93 | $0 / $0 | $-236 / $-236 | $-521 / $-521 | $-993 / $-1,900 | $-2,063 / $-2,328 | $-3,368 / $-4,289 |

## Readout

- This is a first-pass research sweep, not a live candidate.
- The entry assumption is next-open after the extension signal, not guaranteed intrabar limit fill.
- Compare against the main long-only ORB continuation before adding ML.
