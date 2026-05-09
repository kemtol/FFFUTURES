# ORB Feature Research Program

## Overview

Systematic program to discover predictive features for the ORB breakout model's Topstep pass rate. The core insight: the ORB edge existed but faded in 2026 due to regime change (volatility explosion + model OOD). Feature discovery is the path to recovery.

The experiment loop:
1. Propose a feature family with clear hypothesis
2. Implement as a module (zero-touch integration)
3. Generate parquet → auto-detected by module loader
4. Run AB sweep (with/without new module)
5. Keep or discard based on objective criteria
6. Log results and iterate

---

### Data Architecture

```
data/
├── Level_0_Raw/           # Immutable source data
│   ├── MGC_1m.db          # 1-minute OHLCV (SQLite)
│   └── MGC_15m.db         # 15-minute OHLCV (SQLite)
├── Level_1_Features/      # Feature layer (parquet)
│   ├── breakout_events.parquet     # Core breakout events (34,187 rows)
│   ├── market_context.parquet      # Market context (ADX, VWAP, etc.)
│   ├── orb_ranges.parquet          # ORB ranges per session/TF
│   ├── macro_data.parquet          # External macro data (SPX, DXY, US10Y, Oil)
│   └── modules/           # Auto-discovered feature modules
│       ├── orb_context_features.parquet         # Basic orb context
│       ├── scale_invariant_features.parquet     # Normalized features
│       ├── volatility_normalized_features.parquet # Vol-normalized
│       ├── pre_breakout_profile_features.parquet  # Pre-breakout candles
│       ├── session_momentum_features.parquet      # Session momentum
│       ├── interaction_features.parquet           # Interaction terms
│       └── macro_features.parquet                 # Macro regime (SPX/DXY/US10Y/Oil)
└── Level_2_Datamart/      # Training datamart (merged)
    └── training_datamart_orb.parquet
```

### Grain Key (ALL feature modules)

`(date, session, orb_tf, breakout_ts)` — one row per breakout event (34,187 rows).

All modules must use the **same EVENT_KEY** and preserve the **`date` column as `datetime.date` dtype** for correct merge behavior across modules.

### Data Sources Available for Feature Engineering

| Source | Location | Description |
|--------|----------|-------------|
| `breakout_events.parquet` | `data/Level_1_Features/` | Core events: breakout_ts, price, range, ATR, side |
| `market_context.parquet` | `data/Level_1_Features/` | ADX, VWAP distance, ORB/ATR ratio, time_in_session |
| `macro_data.parquet` | `data/Level_1_Features/` | External macro (SPX, DXY, US10Y, Oil daily) |
| `MGC_1m.db` | `data/Level_0_Raw/` | Raw 1m OHLCV with volume (for candle-level features) |
| `MGC_15m.db` | `data/Level_0_Raw/` | Raw 15m OHLCV (for multi-timeframe features) |

### Existing Feature Modules (DO NOT DUPLICATE)

| Module | Features | Source | Status |
|--------|----------|--------|--------|
| `orb_context_features` | 7 basic orb/breakout fields | breakout_events | Active |
| `scale_invariant_features` | 7 normalized variants (z-score, percentile) | breakout_events | Active |
| `volatility_normalized_features` | 5 volatility-normalized | breakout_events | Active |
| `pre_breakout_profile_features` | 5 pre-breakout candle features | MGC_1m.db | Active |
| `session_momentum_features` | 4 session momentum features | MGC_1m.db | Active |
| `interaction_features` | 5 interaction terms | breakout_events + market_context | Active |
| `macro_features` | 4 macro regime features (spx_regime, dxy_trend, us10y_change, oil_volatility) | macro_data.parquet (via yfinance) | Active |

### Evaluation Metrics

- **Score** = `pass_rate - fail_mll_rate` (range -1.0 to +1.0). Higher = better. This is the PRIMARY ranking metric.
- **Pass Rate** = % of trading windows that end with a funded account pass.
- **Fail MLL Rate** = % of windows hitting max losing days (MLL).
- **PnL** = Median ending PnL across windows.
- **2026 Pass Rate** = Most important year-specific metric (hardest regime).

---

### Current Baseline (7-module v6, 2026-04-28)

| Target | Score | Pass | FailMLL | PnL | 2026 Pass |
|--------|:----:|:----:|:-------:|:---:|:---------:|
| `y_1r2_120m` | **+0.5904** | **59.0%** | **0.0%** | **$+2,619** | **70.5%** |
| `y_1r4_120m` | **+0.5301** | **53.0%** | **0.0%** | **$+1,968** | **72.1%** |
| `y_1r2_180m` | **+0.5181** | **51.8%** | **0.0%** | **$+2,133** | **68.9%** |
| `y_1r2_240m` | +0.5181 | 51.8% | 0.0% | $+2,583 | 67.2% |
| `y_1r4_close60m` | +0.4458 | 62.7% | 18.1% | $+3,098 | **73.8%** |
| `y_1r4_240m` | +0.3976 | 39.8% | 0.0% | $+1,983 | 54.1% |
| `y_1r2_60m` | +0.3855 | 61.4% | 22.9% | $+3,262 | 60.7% |
| `y_1r4_180m` | +0.3614 | 36.1% | 0.0% | $+1,835 | 47.5% |
| `y_1r2_close60m` | +0.3133 | 51.8% | 20.5% | $+1,980 | 55.7% |
| `y_1r4_60m` | +0.1446 | 19.3% | 4.8% | $+2,041 | 26.2% |

All 10 targets positive. **7 modules** active: `orb_context`, `scale_invariant`, `volatility_normalized`, `pre_breakout_profile`, `session_momentum`, `interaction`, `macro`.

---

## 2. What You CAN Do

- **Create new feature modules** by copying `_TEMPLATE_generate_feature_module.py` and implementing `load_sources()` + `build_features()`
- **Run AB sweeps**: move module parquet out → sweep → save baseline → restore → sweep → compare
- **Explore SQLite data**: 1m OHLCV with volume, 15m OHLCV without volume
- **Any mathematical transformation**: ratios, z-scores, percentiles, log transforms, interactions, rolling windows, regime flags
- **Cross-timeframe features**: compute from 1m data, merge onto 15m events
- **Fetch external data**: yfinance is now installed — tickers like SPY, DXY (DX-Y.NYB), US10Y (^TNX), Oil (CL=F) are available

## 3. What You CANNOT Do

- **No look-ahead**: features must use only data available BEFORE the breakout timestamp
- **No new raw data fetching**: use only existing MGC_1m.db / MGC_15m.db or already-loaded parquets
- **No label modification**: labels are computed by `build_labels.py`, not by feature modules
- **No modifying existing modules**: create new modules that LEFT JOIN on EVENT_KEY
- **`date` dtype must be `datetime.date`** (not str) for merge compatibility with other modules

---

## 4. Feature Proposal Template

Before implementing a feature, fill out this template:

```markdown
**Feature Name**: `fn_indicator_name`
**Data Source**: [MGC_1m.db | breakout_events | market_context | macro_data.parquet]
**Hypothesis**: [one sentence: "This feature captures..." / "Breakouts with X are more/less likely to pass"]
**Computation**: [brief description of math]
**Expected NaN**: [% estimate — should be <5% for viable features]
```

---

### Implementation Plan

Each feature module follows the same pattern:

1. Copy `_TEMPLATE_generate_feature_module.py` → `generate_{family}_features.py`
2. Update `MODULE_NAME`, docstring with hypothesis
3. Implement `load_sources()` — load needed parquets/SQLite
4. Implement `build_features()` — core computation, return EVENT_KEY + features
5. Run `--dry-run` to verify rows/NaN/conflicts
6. Generate parquet (auto-detected by loader)
7. Run v6 sweep → AB compare
8. Keep/discard, log to `_MEMORY/YYYYMMDD.md`

---

### Allowed Mathematical Transformations

- Ratios (e.g., volume / avg_volume, range / ATR)
- Z-scores (current value relative to rolling distribution)
- Percentile ranks
- Log transforms (for skewed distributions)
- Polynomial interactions (up to 2nd order)
- Binary/ternary flags (above/below threshold)
- Rolling statistics (mean, std, max, min over fixed windows)

### Forbidden Transformations

- Future data (any computation that extends past breakout_ts)
- Trainable parameters (no fitting on the event set)
- Unbounded synthetic expansion (no generating 100+ features from a single signal)

---

## 5. Experiment Loop

```python
# 1. Implement module
cp _TEMPLATE_generate_feature_module.py generate_{family}_features.py
# Edit: MODULE_NAME, load_sources(), build_features()

# 2. Dry-run
python3 generate_{family}_features.py --dry-run
# Expected: rows=34187, NaN<10%, no column conflicts

# 3. Generate parquet
python3 generate_{family}_features.py

# 4. AB comparison protocol
# ── Step A: Save baseline ──
# (If baseline already exists, skip to Step B)
# Move NEW module parquet OUT of modules dir
mv data/Level_1_Features/modules/{family}_features.parquet /tmp/
# Run v6 sweep → baseline results
python3 pipeline/orb_ml/analysis/objective_sweep_orb_v6.py
cp model/SWEEP_v6/OBJECTIVE_SWEEP_RESULTS.csv /tmp/{family}_BASELINE.csv
# Restore NEW module
mv /tmp/{family}_features.parquet data/Level_1_Features/modules/

# ── Step B: Run with new module ──
python3 pipeline/orb_ml/analysis/objective_sweep_orb_v6.py
# Results overwrite OBJECTIVE_SWEEP_RESULTS.csv

# ── Step C: Compare ──
diff /tmp/{family}_BASELINE.csv model/SWEEP_v6/OBJECTIVE_SWEEP_RESULTS.csv
# Or: load both CSVs, compute deltas per target
```

---

### Evaluation Criteria for "Keep"

- Score improves for more targets than it worsens (or has sufficiently large wins to justify regressions)
- Median score change across all 10 targets is positive
- No catastrophic regression (score drop > 0.5) on any target
- 2026 pass rate improves on net (not strictly required but strongly preferred)
- **BUT**: If the hypothesis is sound, the implementation is clean, and the wins are meaningful (even if fewer), the feature can be kept. The goal is discovery, not perfection.

---

## 6. Logging Results

Each experiment is logged to `_MEMORY/YYYYMMDD.md` with:

1. **Hypothesis** — what we expected to find
2. **Implementation** — module name, features, data sources
3. **AB Comparison** — table of deltas per target
4. **Scoreboard** — the new baseline after decision
5. **Decision** — keep or discard, with reasoning
6. **Next steps** — what to try next

---

## 7. Feature Family Ideas (Starting Points)

These are ordered by estimated effort-to-impact ratio:

### ✅ Priority A: Parameter Tuning (zero code, re-sweep only) — DONE

Higher rev_q (0.85-0.95) reduces fail-MLL 20-35pp but insufficient for GO/NO-GO.

### ✅ Cycle 1: Pre-Breakout Volatility Profile — KEPT

Implemented as [`pre_breakout_profile_features`](../pipeline/orb_ml/features/modules/generate_pre_breakout_profile_features.py) (5 features from 4×15m pre-breakout candles).

**Results**: All 10 targets improved. 2R targets revived from 0% to 27.7-49.4% pass. 4R targets flipped from negative scores to positive. 2026 pass rates 34-70%.

### ✅ Cycle 2: Session Momentum — KEPT (2026-04-28)

Implemented as [`session_momentum_features`](../pipeline/orb_ml/features/modules/generate_session_momentum_features.py) (4 features from 1m OHLCV).

**Results**: y_1r4_180m reached 93.4% pass in 2026 ($4,089 PnL) — single best result ever. 6/10 targets improved, 3 regressed. KEPT over strict criteria due to magnitude of wins.

### ✅ Cycle 3: Interaction Features — KEPT (2026-04-28)

Implemented as [`interaction_features`](../pipeline/orb_ml/features/modules/generate_interaction_features.py) (5 features from breakout_events × market_context).

**Results**: 4/10 targets improved substantially (especially short-duration: y_1r2_120m +0.5060, y_1r4_60m +0.1205). Long-duration targets regressed (y_1r4_180m -0.3614). KEPT — interaction terms add meaningful signal for short holds with zero data cost.

### ✅ Cycle 4: Macro Features — KEPT (2026-04-28)

Implemented as [`macro_features`](../pipeline/orb_ml/features/modules/generate_macro_features.py) (4 features from macro_data.parquet: SPX regime, DXY trend, US10Y change, Oil volatility).

**Results**: 4/10 targets improved (y_1r4_240m +0.2530, y_1r4_close60m +0.1446, y_1r2_60m +0.1325, y_1r4_120m +0.1084). Net Δ score +0.1928. **Complementary pattern to interaction features**: macro helps LONG durations, interaction helps SHORT durations. Key wins: y_1r4_close60m 73.8% 2026 pass ($3,098 PnL), y_1r4_240m 54.1% 2026 pass (+23pp). KEPT.

### ✅ Cycle 5: Regime Features — DISCARDED (2026-04-28)

7 features: vol regime, ADX regime, session persistence, ORB range CV. 10/10 targets regressed (net Δ -1.43). **Lesson**: LGBM trees are already implicit regime detectors given 42 features.

### ✅ Cycle 6: Walk-Forward Training v2.0 — BREAKTHROUGH (2026-04-28)

All 42 features combined. Walk-forward 2020-2026. ALL 10 targets positive across ALL 7 years. 2026 positive for every target (0.48R–1.24R). 4R targets lead: y_1r4_180m +1.22R, y_1r4_240m +1.22R.

### ✅ Cycle 7: Feature Importance (2026-04-28)

Pre-breakout profile dominates (28.5%), followed by orb_context (18.1%), scale_invariant (13.3%), session_momentum (12.8%), interaction (10.8%), vol_norm (6.6%), macro (5.9%).

### ✅ Cycle 7b: Hyperparameter Tuning — SKIPPED (2026-04-28)

Random search 40 combos showed no improvement vs default LGBM params (lr=0.05, leaves=31). Default already near-optimal.

### 🏆 Cycle 8: Topstep 50K Pass-Rate — GO (2026-04-28)

Holdout 2025-12-01 → 2026-04-24. 128 policy combos per target. **88% pass, 0% fail MLL**. ORB_v2.0 is a sim GO candidate; live validation is still pending.

### 🔄 Phase 6: Live Inference & Execution (2026-04-29 → now)

Live daemon running on TopstepX eval account:
- TopstepX real-time 1m OHLCV via WebSocket
- Feature builder replicating batch pipeline (42 features)
- LGBM inference with policy filter (rev/cont/skip)
- TopstepX REST order execution with browser-mimic headers
- Telegram bot for monitoring (/status, /last, /pnl, /commands)
- Paper portfolio tracker with TP/SL auto-close

**Calibration progress**:
- Batch/live feature replication is mostly solved at module level: core, scale-invariant, volatility-normalized, session-momentum, and interaction are 100% within `<0.01`; pre-breakout is 98.6%, orb-context 97.3%, macro 90.5%.
- TradingView ST/DEMA/ADX/CCI comparator is calibrated separately: 32 closed trades matched, 1 open trade entry matched, 96.97% match.
- Databento Apr 16-Apr 30, 2026 repair fixed bad local OHLC candles that were causing comparator mismatches.

**Current blocker (2026-05-02)**:
- Live runner is operational and heartbeating.
- Persisted paper/live portfolio has 8 closed ORB_v2 trades, all stopped out, balance $49,176 from $50,000.
- This live loss streak conflicts with the Topstep sim profile. Next work is event-level replay: exact live timestamps, source candles, features, model probabilities, decision, and TP/SL compared to batch-equivalent output.

---

## 8. Failure Modes

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| pass_rate stays 0% for 2R targets | 2R not viable for this strategy | Accept; focus on 4R |
| pass_rate improves but fail_mll doubles | Feature increases trade frequency too much | Try higher rev_q, or combine with volatility filter |
| pass_2026 improves but pre-2025 collapses | Feature overfits to 2026 volatility regime | Check if feature is regime-dependent |
| All pass rates go down | Feature adds noise, not signal | Discard immediately |
| Generator OOM or too slow | Window computation too large | Optimize rolling window, reduce history |
| Column name conflict | Feature name already exists | Rename with prefix or check existing modules |
| 100% NaN on all features after merge | date dtype mismatch (str vs datetime.date) | Convert to `datetime.date` in load_sources() |
| All labels fail "Too few training rows: train=0" | Merge produced NaN on all feature cols | Check date dtype compatibility in parquet |

---

## 9. Quick Reference

```bash
# Create new module (copy template)
cp pipeline/orb_ml/features/modules/_TEMPLATE_generate_feature_module.py \
   pipeline/orb_ml/features/modules/generate_{family}_features.py

# Dry-run
python3 pipeline/orb_ml/features/modules/generate_{family}_features.py --dry-run

# Generate (with force if conflicts are intentional)
python3 pipeline/orb_ml/features/modules/generate_{family}_features.py [--force]

# Run sweep
python3 pipeline/orb_ml/analysis/objective_sweep_orb_v6.py

# Read results
cat model/SWEEP_v6/OBJECTIVE_SWEEP_RESULTS.csv
```
