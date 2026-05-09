# MMMACHINE Futures Research: Gemini CLI Context

This project is a quantitative research repository focused on identifying and validating Opening Range Breakout (ORB) strategies for MGC (Micro Gold Futures) and maintaining a live auto-trade pipeline via TopstepX.

## Project Overview (Dual-Track)

1.  **ORB ML Research (Track A):** Identify an edge that passes the Topstep 50K evaluation ($3,000 target, $2,000 MLL, 50% consistency) in ~20 trading days using LightGBM models.
2.  **Live Auto-Trade Pipeline (Track B):** A trend-following strategy (`Super Structure`) ported from TradingView (ST+DEMA+ADX+CCI) running as a systemd daemon, executing trades on TopstepX and reporting via Telegram.

## Core Architecture & Data Layers

| Layer | Path | Description |
| :--- | :--- | :--- |
| **Level 0 (Raw)** | `data/Level_0_Raw/` | Immutable SQLite databases (`MGC_1m.db`, etc.). |
| **Level 1 (Events)** | `data/Level_1_Features/` | Parquet files for events, market context, and macro data. |
| **Level 2 (Datamart)** | `data/Level_2_Datamart/` | `training_datamart_orb.parquet` (68k+ rows). |
| **Live State** | `data/Live/` | `combined_buffer.db` (3.6M bars), tokens, logs, and signals. |
| **Models** | `model/ORB_v2.0_.../` | Active LightGBM models and evaluation reports. |

## Building and Running

The project uses script-based execution from the project root.

### Common Research Workflow (Track A)
```bash
# Generate a new feature module from template
cp pipeline/orb_ml/features/modules/_TEMPLATE_generate_feature_module.py pipeline/orb_ml/features/modules/generate_NEW_features.py
python3 pipeline/orb_ml/features/modules/generate_NEW_features.py --force

# Run objective sweep (Train + Sim)
python3 pipeline/orb_ml/analysis/objective_sweep_orb_v6.py
```

### Live Trading Ops (Track B)
```bash
# Manage the Super Structure daemon
systemctl --user {start|stop|restart|status} super_structure
journalctl --user -u super_structure -f

# Manage the TopstepX WebSocket feed
systemctl --user {start|stop|restart|status} topstepx_feed
journalctl --user -u topstepx_feed -f
```

## Development Conventions

### Data Integrity & Types
- **Date Format:** ALL `date` columns MUST be `datetime.date`. String dates cause silent `NaN` on joins.
- **Merge Grain:** Modules join on `(date, session, orb_tf, breakout_ts)`.
- **No Look-ahead:** Features must only use data available *before* the event.

### Simulation & Execution Rules
- **Metric of Success:** `score = pass_rate - fail_mll_rate`.
- **Trading Day:** CT-based. Use `map_to_topstep_trade_day()` (subtracts 15h10m from UTC).
- **Commissions:** $3.00/RT for ORB research; **$1.74/RT** for Super Structure live auto-trade.
- **Resampling:** MUST use `label="right", closed="left"` for 5m bars to match TradingView timestamps.

## Critical Footguns & Gotchas

- **No State Persistence:** `SuperStructure` state (position, SL) resets on restart. If the daemon restarts while a trade is open, it may double-trade or lose track of the exit.
- **No Exchange Stop Orders:** SL is code-managed via SuperTrend trailing. If the daemon dies, the position has zero protection.
- **Duplicate Exit Logic:** `super_structure.py` contains redundant exit check blocks (lines 392-418 and 420-435). Edit with caution.
- **Warmup Requirement:** DEMA(200) requires ≥120 trading days of warmup for convergence.
- **Single Instance:** Never run `restart_daemons.sh` and systemd concurrently; it results in double execution.
- **ORB v2.0 is Research-Only:** It is NOT wired for live execution or Telegram reporting.

## Active Research State
As of May 2026, the project is resolving a 2026 OOD failure using scale-invariant and volatility-normalized features. The `Super Structure` live pipeline is currently operational via systemd.
