# MNQ ORB Rule-Based 15m Long TP2R/EOD

Dedicated artifact folder for the current MNQ rule-based candidate.

## Files

| File | Description |
| --- | --- |
| `events.parquet` | One row per executed strategy trade, net of TopstepX MNQ fee and modeled slippage |
| `summary.json` | Machine-readable performance summary |
| `manifest.json` | Build metadata and source references |
| `report.md` | Human-readable strategy report |
| `flash_guard_sweep.csv` | Catastrophic safety guard grid, separate from normal strategy exits |
| `flash_guard_report.md` | Human-readable flash guard report |
| `README.md` | This folder guide |

## Current Snapshot

| Metric | Value |
| --- | ---: |
| Trades | 1,296 |
| Win rate | 56.48% |
| Net PnL | $33,091 |
| Max DD | $-12,124 |
| Profit factor | 1.12 |
| Daily Sharpe | 0.50 |
| Daily Sortino | 0.64 |

## Contract

- 15m opening range.
- Long only after close above OR high.
- Entry on next M1 open.
- Exit on TP 2R or 15:00 NY time exit.
- No normal strategy SL; OR low is sizing reference only.
- Costs include TopstepX MNQ $1.24 round-turn per contract plus 1 tick slippage per side.

## Flash Guard

The base strategy still exits only by TP 2R or 15:00 NY. A separate catastrophic
guard sweep is available in `flash_guard_report.md` for live risk planning.
