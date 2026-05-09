# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 3.12 research repo for ORB (Opening Range Breakout) strategy on MGC Micro Gold Futures. The goal is finding an edge that passes a Topstep 50K evaluation in ~20 trading days ($3,000 profit target, $2,000 MLL, consistency rule). See `README.md` for full research state and `AGENTS.md` for the complete architecture reference.

No build system, no test framework, no CI. All scripts run with `python3` from the project root.

## Primary Commands

### Main Research Loop (feature module → sweep)

```bash
# Create a new feature module from template
cp pipeline/orb_ml/features/modules/_TEMPLATE_generate_feature_module.py \
   pipeline/orb_ml/features/modules/generate_{family}_features.py

# Dry-run before writing
python3 pipeline/orb_ml/features/modules/generate_{family}_features.py --dry-run

# Generate parquet
python3 pipeline/orb_ml/features/modules/generate_{family}_features.py [--force]

# Run v6 sweep (auto-discovers all modules in data/Level_1_Features/modules/)
python3 pipeline/orb_ml/analysis/objective_sweep_orb_v6.py
```

### Training & Evaluation

```bash
python3 pipeline/orb_ml/train/train_orb_reversal.py
python3 pipeline/orb_ml/train/train_orb_continuation.py
python3 pipeline/orb_ml/train/train_orb_walk_forward_v2.py   # Walk-forward (all 42 features)
python3 pipeline/orb_ml/analysis/eval_holdout_orb.py
python3 pipeline/orb_ml/analysis/eval_policy_switch_orb.py
python3 pipeline/analysis/eval_topstep_pass_v2.py     # Topstep 50K pass-rate (v2.0 models)
python3 pipeline/analysis/test_refined_sim.py
```

### Data Pipeline (run in order when rebuilding from raw)

```bash
bash pipeline/run/run_fetch_mgc.sh
python3 pipeline/fetch/fetch_macro_data.py
python3 pipeline/orb_ml/features/build_orb_ranges.py
python3 pipeline/orb_ml/features/build_breakout_events.py
python3 pipeline/orb_ml/features/build_market_context.py
python3 pipeline/orb_ml/features/build_labels.py
# Then regenerate all feature modules
```

## Architecture

### Data Layers

| Path | Contents | Notes |
|------|----------|-------|
| `data/Level_0_Raw/` | Immutable SQLite: MGC_1m.db, MGC_5m.db, MGC_15m.db | Source of truth |
| `data/Level_1_Features/` | Parquet: breakout_events (34,187 rows), market_context, orb_ranges, macro_data | Event-grain |
| `data/Level_1_Features/modules/` | Auto-discovered `*_features.parquet` files | One per feature family |
| `data/Level_2_Datamart/` | training_datamart_orb.parquet (68,374 rows = 2× breakout events: rev + cont) | Model input |
| `data/Live/` | Live trading state: SQLite buffers, JWT token, Telegram config | Not in research loop |

### Pipeline

- **`pipeline/orb_ml/features/modules/`** — Feature generators (standalone CLIs) + `loader.py` (auto-joins all modules by filename sort) + `_TEMPLATE_generate_feature_module.py`
- **`pipeline/analysis/topstep_sim.py`** — Shared Topstep simulator module. The baseline for all simulation work.
- **`pipeline/orb_ml/analysis/objective_sweep_orb_v6.py`** — Active sweep runner. Trains rev+cont LGBM models per label, simulates Topstep, reports pass/fail scores.
- **`pipeline/orb_ml/train/`** — LGBM trainers (reversal, continuation, walk-forward v2)
- **`pipeline/live/`** — Live inference daemon (not in research loop; see calibration state in `RETROSPECTIVE.md`)

### Models

- Active: `model/ORB_v2.0_2010-2026/` — 20 models (rev+cont × 10 labels × 5 hold periods)
- Baseline: `model/ORB_v1.0/` — original rev/cont pair (120m target)
- Model filenames: `lgbm_{rev|cont}_v2_y_{ratio}_{horizon}.txt`

## Critical Invariants

- **`date` dtype must be `datetime.date`**, not `str`. String dates cause silent NaN on all feature columns after merge. Every module must do `df["date"] = pd.to_datetime(df["date"]).dt.date`.
- **Module grain**: `(date, session, orb_tf, breakout_ts)` — one row per breakout event (34,187 rows). Do NOT include `year` or `side` in merge keys. The loader LEFT JOINs onto the 2-row rev/cont datamart automatically.
- **No look-ahead**: features must only use data available before `breakout_ts`.
- **Column conflicts**: two modules with the same column name get pandas `_x`/`_y` suffixes silently. The loader warns; avoid duplicate column names across modules.
- **Data leakage**: `TRAIN_TO="2025-11-30"`, `HOLDOUT_FROM="2025-12-01"` — always a gap.
- **Topstep trading day**: CT-based via `map_to_topstep_trade_day()` (subtracts 15h10m from UTC).
- **Commission**: $3.00/round-turn (fixed). MGC = $1.00/tick/contract.
- **Macro features** merge on `date` only (not full event key) and are forward-filled for weekends/holidays.

## Scoring Metric

**`score = pass_rate - fail_mll_rate`** (range −1.0 to +1.0). This is the primary ranking metric — not AUC, not win rate. 2026 pass rate is the most important year-specific metric (hardest out-of-distribution regime). Optimize for score; report 2026 pass_rate separately.

## Active Feature Modules (7 modules, 42 features)

| Module file | Features | Key signal |
|-------------|:--------:|------------|
| `generate_orb_context_features.py` | 7 | VWAP dist, ADX, EMA slope, day-of-week |
| `generate_scale_invariant_features.py` | 7 | Ratios + squares robust to volatility scaling |
| `generate_volatility_normalized_features.py` | 5 | Rolling percentile ranks and z-scores |
| `generate_pre_breakout_profile_features.py` | 5 | Compression, drift, inside-bar (most predictive family) |
| `generate_session_momentum_features.py` | 4 | First-30m range/direction, volume ratio/z-score |
| `generate_interaction_features.py` | 5 | Cross-product terms (ATR×ADX, strength×range, etc.) |
| `generate_macro_features.py` | 4 | SPX regime, DXY trend, US10Y change, oil vol |

## What NOT to Do

- Do not use `inferences/orb/predict.py` — stale, references wrong target names and model paths.
- Do not use `edges/orb_breakout/` — legacy naming conventions, incompatible grain.
- Do not use `_ARCH/` — dead experiments (FastAPI, vectorbt, old strategies).
- Do not add tests — no test framework. `test_refined_sim.py` is an ad-hoc comparison, not a test suite.
- Do not modify existing feature modules — always create new ones (or use `--force` to regenerate).
- Do not add `year` to merge keys.
- Do not optimize for AUC. Use `score = pass_rate - fail_mll_rate`.
- Do not claim any model is trade-ready until consistent 2026 pass rate is demonstrated.

## Dependencies

**`requirements.txt`**: `aiohttp`, `pandas`, `pyarrow`, `playwright`, `websockets`, `yfinance`, `databento`, `zstandard`

**Implicit** (not in requirements.txt): `lightgbm`, `numpy`, `scikit-learn`
