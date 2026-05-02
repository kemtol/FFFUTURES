# AGENTS.md — MGC ORB Futures Research

Python 3.12 research repo. Goal: find an ORB-based edge on MGC Micro Gold futures that passes a Topstep 50K evaluation in ~20 trading days. No formal build/test/lint/CI tooling. All scripts run with `python3` from project root.

## Architecture

```
data/
├── Level_0_Raw/         # Immutable SQLite: MGC_1m.db, MGC_5m.db, MGC_15m.db
├── Level_1_Features/    # Parquet: breakout_events (34,187 rows), market_context, orb_ranges, macro_data
│   └── modules/         # AUTO-DISCOVERED feature parquets (*_features.parquet)
├── Level_2_Datamart/    # training_datamart_orb.parquet (68,374 rows = 2× breakout events, rev+cont)
└── Live/                # Live trading state (config + SQLite buffer + daemon log)
    ├── topstepx_buffer.db    # SQLite: ohlcv_1m (TopstepX), ohlcv_1m_yfinance (fallback)
    ├── topstepx_token.json   # JWT for TopstepX API
    ├── topstepx_creds.json   # TopstepX email + password
    └── telegram.env          # Telegram bot token + chat ID
pipeline/
├── live/                # 🔴 LIVE INFERENCE & EXECUTION (not in research loop)
│   ├── runner.py           # Main daemon: ORB detect → features → predict → execute → Telegram
│   ├── feature_builder.py  # Replicates batch 42-feature pipeline from buffer
│   ├── buffer.py           # SQLite buffer manager (TopstepX WS → SQLite)
│   ├── orb_detector.py     # Session management, ORB range, breakout detection
│   ├── portfolio.py        # Paper trading tracker with TP/SL auto-close
│   ├── bot.py              # Telegram bot: /status, /last, /pnl, /portfolio, /features
│   ├── rolling.py          # RollingStats manager (skeleton, not yet integrated)
│   ├── execute/
│   │   └── topstepx.py     # TopstepX REST order execution (browser-mimic headers)
│   └── sources/
│       └── topstepx.py     # TopstepX WebSocket feed (RealTimeBar, SubscribeBars)
├── feature/modules/     # Feature generators (standalone CLI) + loader.py + _TEMPLATE
├── analysis/            # Sweep runners, topstep_sim.py (shared), evaluation
├── train/               # LGBM trainers (reversal + continuation)
└── fetch/               # yfinance data ingestion
model/                   # Trained models + sweep reports
_MEMORY/                 # Daily research logs (agent continuity)
inferences/orb/          # STALE — do not use
edges/orb_breakout/      # LEGACY — different target names, no Topstep sim
_ARCH/                   # DEAD CODE — FastAPI backtest, old strategies
ui/                      # Lightweight Charts SPA (JS/HTML)
service/                 # EMPTY
```

## Critical Gotchas (miss these and everything breaks)

- **`date` dtype must be `datetime.date`**, not `str`. String dates cause silent NaN on ALL feature columns after merge. Every module must assert `df["date"] = pd.to_datetime(df["date"]).dt.date`.
- **Module grain**: `(date, session, orb_tf, breakout_ts)` — one row per breakout event (34,187 rows). Do NOT include `year` or `side` in merge keys. The loader LEFT JOINs onto the 2-row rev/cont datamart.
- **No look-ahead**: features must use only data available BEFORE `breakout_ts`.
- **Alphabetical merge order**: `loader.py` merges modules sorted by filename. Adding a module whose name sorts earlier shifts merge order (should be harmless, but be aware).
- **Column name conflicts**: if two modules define the same feature column, pandas adds `_x`/`_y` suffixes silently. The loader warns. Use `--force` when regenerating a module with intent.
- **Topstep trading day**: US CT-based. `map_to_topstep_trade_day()` subtracts 15h10m to map UTC timestamps to 5PM CT → 3:10PM CT next day boundary.
- **Fixed commission**: $3.00/round-turn (not percentage). MGC = $1.00/tick/contract.
- **Data leakage**: `TRAIN_TO` must end before `HOLDOUT_FROM`. Set `TRAIN_TO="2025-11-30"`, `HOLDOUT_FROM="2025-12-01"`.
- **Macro features merge on `date` only** (not full EVENT_KEY), forward-fill for weekends.
- **All 10 labels are in the datamart**: `y_1r2_60m`, `y_1r4_60m`, `y_1r2_120m`, `y_1r4_120m`, `y_1r2_180m`, `y_1r4_180m`, `y_1r2_240m`, `y_1r4_240m`, `y_1r2_close60m`, `y_1r4_close60m`.

## Dependencies

**In requirements.txt** (8 deps): `aiohttp`, `pandas`, `pyarrow`, `playwright`, `websockets`, `yfinance`, `databento`, `zstandard`

**Implicit** (used in code, NOT in requirements.txt): `lightgbm`, `numpy`, `scikit-learn` (edges/ only)

**yfinance** is installed with `--break-system-packages`. Available tickers: SPY, DX-Y.NYB, ^TNX, CL=F.

## Primary Commands

### Feature Module Workflow (the main research loop)

```bash
# Create new module from template
cp pipeline/feature/modules/_TEMPLATE_generate_feature_module.py \
   pipeline/feature/modules/generate_{family}_features.py

# Dry-run (verifies rows, NaN%, conflicts before writing)
python3 pipeline/feature/modules/generate_{family}_features.py --dry-run

# Generate parquet
python3 pipeline/feature/modules/generate_{family}_features.py [--force]

# Run active sweep (v6 modular — auto-discovers all modules)
python3 pipeline/analysis/objective_sweep_orb_v6.py

# AB comparison: move module out → sweep (baseline) → move back → sweep (test) → diff CSVs
```

### Regenerate existing modules

```bash
python3 pipeline/feature/modules/generate_orb_context_features.py
python3 pipeline/feature/modules/generate_scale_invariant_features.py
python3 pipeline/feature/modules/generate_volatility_normalized_features.py
python3 pipeline/feature/modules/generate_pre_breakout_profile_features.py
python3 pipeline/feature/modules/generate_session_momentum_features.py
python3 pipeline/feature/modules/generate_interaction_features.py
python3 pipeline/feature/modules/generate_macro_features.py
```

### Training & Evaluation

```bash
python3 pipeline/train/train_orb_reversal.py
python3 pipeline/train/train_orb_continuation.py
python3 pipeline/train/train_orb_walk_forward_v2.py     # Walk-forward v2.0 (all 42 features)
python3 pipeline/analysis/eval_holdout_orb.py
python3 pipeline/analysis/eval_policy_switch_orb.py
python3 pipeline/analysis/plot_policy_pnl_state.py
python3 pipeline/analysis/test_refined_sim.py
python3 pipeline/analysis/eval_topstep_pass_v2.py       # Topstep 50K pass-rate (v2.0 models)
```

### Data Fetching

```bash
bash pipeline/run/run_fetch_mgc.sh
python3 pipeline/fetch/fetch_macro_data.py
python3 pipeline/feature/build_orb_ranges.py
python3 pipeline/feature/build_breakout_events.py
python3 pipeline/feature/build_market_context.py
python3 pipeline/feature/build_labels.py
```

## Scoring Metric

**`score = pass_rate - fail_mll_rate`** (range -1.0 to +1.0). This is the PRIMARY ranking metric, not AUC or win rate. 2026 pass rate is the most important year-specific metric (hardest regime).

## Session Convention

| Session | UTC Open | UTC Close | ORB Complete |
|---------|:-------:|:---------:|:-----------:|
| Tokyo   | 00:00   | 03:00     | 00:15        |
| London  | 07:00   | 10:00     | 07:15        |
| US      | 13:30   | 16:30     | 13:45        |

## ORB Timeframes

| Label   | Source DB   | Candles | Period  |
|---------|-------------|:-------:|:-------:|
| ORB-5m  | MGC_5m.db   |    3    | 15 min  |
| ORB-15m | MGC_15m.db  |    1    | 15 min  |
| ORB-30m | MGC_15m.db  |    2    | 30 min  |

## What NOT to do

- Do not claim ORB_v1.0 is trade-ready.
- Do not use `inferences/orb/predict.py` — stale, references wrong target names and model paths.
- Do not use code in `edges/orb_breakout/` — legacy naming conventions, incompatible grain.
- Do not use code in `_ARCH/` — dead experiments (FastAPI, vectorbt, old strategies).
- Do not create/run tests — no test framework exists. `test_refined_sim.py` is an ad-hoc comparison, not a test suite.
- Do not modify existing feature modules — always create new ones.
- Do not add `year` to EVENT_KEY merge keys.
