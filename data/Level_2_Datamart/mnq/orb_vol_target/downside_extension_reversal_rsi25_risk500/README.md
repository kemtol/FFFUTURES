# MNQ ORB Downside Extension Long Reversal Sweep

Created: `2026-05-30T23:29:27.852776+00:00`

## Contract

- Long-only reversal after downside extension below the 15m opening range.
- Entry modes: `touch_next_open` and `close_next_open`.
- TP is whichever is hit first: dynamic session VWAP or fixed +2R from entry.
- SL is one OR range below entry for the default sweep.
- Same-bar SL/TP ambiguity is resolved conservatively: SL first.
- RSI filter: `RSI14 <= 25`.

## Top Rows

| Rank | Ext R | Entry | RSI Max | Risk | Trades | Win Rate | PnL | Max DD | R/DD | 5D | 10D | 20D | 50D | 100D | 200D |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | `close_next_open` | 25 | $500 | 136 | 38.2% | $-5,747 | $-9,704 | -0.59 | $0 / $0 | $0 / $0 | $0 / $0 | $829 / $0 | $2,169 / $0 | $1,756 / $-785 |
| 2 | 1.5 | `close_next_open` | 25 | $500 | 182 | 38.5% | $-7,791 | $-11,324 | -0.69 | $0 / $0 | $0 / $0 | $0 / $0 | $464 / $0 | $577 / $-470 | $332 / $-956 |
| 3 | 2 | `touch_next_open` | 25 | $500 | 155 | 41.3% | $-3,821 | $-7,051 | -0.54 | $0 / $0 | $0 / $0 | $0 / $0 | $0 / $0 | $812 / $-470 | $345 / $-937 |
| 4 | 1.5 | `touch_next_open` | 25 | $500 | 192 | 37.0% | $-12,405 | $-14,179 | -0.87 | $0 / $0 | $0 / $0 | $0 / $0 | $464 / $0 | $580 / $-470 | $11 / $-1,206 |
| 5 | 1 | `close_next_open` | 25 | $500 | 242 | 39.3% | $-16,570 | $-19,011 | -0.87 | $0 / $0 | $0 / $0 | $0 / $0 | $-963 / $-963 | $-1,054 / $-1,525 | $-3,148 / $-3,508 |
| 6 | 1 | `touch_next_open` | 25 | $500 | 252 | 40.5% | $-17,186 | $-19,809 | -0.87 | $0 / $0 | $0 / $0 | $0 / $0 | $-1,038 / $-1,379 | $-1,880 / $-1,880 | $-4,417 / $-4,417 |

## Readout

- This is a first-pass research sweep, not a live candidate.
- The entry assumption is next-open after the extension signal, not guaranteed intrabar limit fill.
- Compare against the main long-only ORB continuation before adding ML.
