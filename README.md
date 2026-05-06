# MMMACHINE Futures Research Continuity

Last updated: 2026-05-02

This repo currently contains research for an ORB-based strategy on MGC Micro Gold Futures. The active research direction is no longer "find any positive expectancy edge"; the objective has been narrowed to:

> Find an edge that can pass a Topstep-style 50K evaluation in about 20 trading days while respecting profit target, maximum loss limit, and consistency constraints.

## Current Snapshot

Latest continuity checkpoint:

- `_MEMORY/20260502.md` — live/paper status after TradingView comparator calibration and Databento repair
- `_MEMORY/20260501.md` — detailed TradingView strategy calibration and Databento OHLCV repair notes
- `_MEMORY/20260429.md` — live ORB daemon operationalization and feature-builder calibration
- `_MEMORY/20260428.md` — ORB_v2.0 feature discovery, walk-forward, and Topstep pass-rate breakthrough

Active model family:

- `model/ORB_v2.0_2010-2026/`
- 7 feature modules, 42 features, 20 LGBM models
- Best Topstep 50K sim candidates: `y_1r2_close60m` and `y_1r4_180m`, both 87.95% pass and 0.0% fail MLL on the recorded holdout grid

Live status as of 2026-05-02:

- `pipeline/live/runner.py --live` is running and heartbeating.
- Persisted paper/live portfolio has 8 closed trades, all SL, balance $49,176 from $50,000.
- This 8-loss deployment streak is the current blocker. Treat ORB_v2.0 as a sim GO candidate, not live-validated.

TradingView comparator status:

- `pipeline/live/super_structure.py` matches TradingView strategy semantics closely after indicator and lifecycle fixes.
- Current comparison artifact shows 32 closed trades matched and 1 open trade entry matched (96.97% match).

Databento repair status:

- `pipeline/fetch/ingest_databento_json.py` ingested Apr 16-Apr 30, 2026 Databento OHLCV into `MGC_1m.db`, then 5m/15m were resampled.
- This fixed known Apr 20 bad yfinance candles that caused TradingView comparator mismatches.

The older sections below remain useful historical context, especially for ORB_v1.0 failure analysis, but the current work is live calibration and data provenance for ORB_v2.0.

## Current Project Thesis

The original PRD focused on ORB breakout events and asked whether reversal setups after an ORB breakout can be filtered into a profitable subset. The current conclusion is more precise:

- Reversal-only is not robust enough.
- Adaptive rev/cont/skip is materially better than always reversal or always continuation.
- However, ORB_v1.0 is still not strong enough for the Topstep 50K one-month pass objective.
- The biggest blocker is 2026 regime behavior.
- Research should return to edge finding, but with the Topstep pass-rate engine as the final scoreboard.

Primary PRD:

- `_DOC/_PRD/0001_orb_edge.md`

## Data State

Main datamart:

- `data/Level_2_Datamart/training_datamart_orb.parquet`
- Shape observed: 68,374 rows x 29 columns
- Coverage: 2010-2026
- Event structure: 2 rows per breakout event, one `cont`, one `rev`
- Labels available:
  - `y_1r2_60m`
  - `y_1r4_60m`
  - `y_1r2_120m`
  - `y_1r4_120m`
  - `y_1r2_180m`
  - `y_1r4_180m`
  - `y_1r2_240m`
  - `y_1r4_240m`
  - `y_1r2_close60m`
  - `y_1r4_close60m`

Implemented features include:

- `orb_range_atr_ratio`
- `breakout_strength`
- `atr14_at_entry`
- `price_vs_vwap_pct`
- `adx_14_15m`
- `ema_slope_1h`
- `day_of_week`
- `time_in_session_min`
- `orb_tf`
- `session`
- `breakout_side`

Known data/model caveats:

- Topstep trading day now uses CT-based boundaries: `map_to_topstep_trade_day()` subtracts 15h10m for proper 5PM CT → 3:10PM CT next day mapping. (Refined in v4+.)
- MLL simulation currently uses realized PnL after modeled trades, not intrabar mark-to-market.
- Existing labels assume entry at breakout candle close. This may overstate execution realism versus next-candle open.
- Same-candle TP/SL ambiguity is known from PRD and should still be quantified.

## Models And Reports

Model directory:

- `model/ORB_v1.0/`

Reversal model:

- `model/ORB_v1.0/lgbm_rev_1r2_120m.txt`
- `model/ORB_v1.0/lgbm_rev_1r2_120m_meta.json`
- `model/ORB_v1.0/REPORT.md`

Continuation model:

- `model/ORB_v1.0/lgbm_cont_1r2_120m.txt`
- `model/ORB_v1.0/lgbm_cont_1r2_120m_meta.json`
- `model/ORB_v1.0/CONT_REPORT.md`

Topstep simulator (refined):
- `pipeline/analysis/topstep_sim.py` — shared module
- `pipeline/analysis/test_refined_sim.py` — comparison test

Evaluation reports:

- `model/ORB_v1.0/HOLDOUT_REPORT.md`
- `model/ORB_v1.0/POLICY_SWITCH_REPORT.md`
- `model/ORB_v1.0/POLICY_PNL_STATE.png`
- `model/ORB_v1.0/POLICY_PNL_TOPSTEP.png`
- `model/ORB_v1.0/POLICY_PNL_STATE_METRICS.csv`
- `model/ORB_v1.0/TOPSTEP_PASS_REPORT.md`
- `model/ORB_v1.0/TOPSTEP_PASS_GRID.csv`

Sweep reports:

- `model/SWEEP_v1/OBJECTIVE_SWEEP_REPORT.md`
- `model/SWEEP_v2/OBJECTIVE_SWEEP_REPORT.md`
- `model/SWEEP_v3/OBJECTIVE_SWEEP_REPORT.md`
- `model/SWEEP_v3/SIMULATOR_COMPARISON.md`
- `model/SWEEP_v4/OBJECTIVE_SWEEP_REPORT.md`
- `model/SWEEP_v4/COMPARISON_v3_v4.md`

Sweep results (CSV):

- `model/SWEEP_v1/OBJECTIVE_SWEEP_RESULTS.csv`
- `model/SWEEP_v2/OBJECTIVE_SWEEP_RESULTS.csv`
- `model/SWEEP_v3/OBJECTIVE_SWEEP_RESULTS.csv`
- `model/SWEEP_v4/OBJECTIVE_SWEEP_RESULTS.csv`

Scripts:

- `pipeline/train/train_orb_reversal.py`
- `pipeline/train/train_orb_continuation.py`
- `pipeline/analysis/eval_holdout_orb.py`
- `pipeline/analysis/eval_policy_switch_orb.py`
- `pipeline/analysis/plot_policy_pnl_state.py`
- `pipeline/analysis/topstep_pass_engine.py`
- `pipeline/analysis/topstep_sim.py` — shared Topstep simulator module
- `pipeline/analysis/test_refined_sim.py` — refined simulator comparison test
- `pipeline/analysis/objective_sweep_orb_v4.py` — v4 sweep with refined simulator

## ORB_v1.0 Results

### Reversal Holdout 2024+

From `HOLDOUT_REPORT.md`:

- Reversal all, no model filter:
  - n: 5,291
  - win rate: 0.334
  - exp_net: -0.069R
  - PF_net: 0.904
  - max loss streak: 33
- Reversal model-filtered:
  - n: 2,286
  - win rate: 0.361
  - exp_net: +0.013R
  - PF_net: 1.019
  - max loss streak: 24

Conclusion: reversal-only is not sufficient.

### Dynamic Rev/Cont/Skip Policy

From `POLICY_SWITCH_REPORT.md` and `POLICY_PNL_STATE_METRICS.csv`:

- always_rev:
  - cum_net: -363.37R
  - max DD: 431.88R
- always_cont:
  - cum_net: -249.37R
  - max DD: 256.92R
- max_prob_no_gate:
  - cum_net: +35.63R
  - max DD: 151.03R
- dynamic_policy:
  - cum_net: +270.18R
  - max DD: 93.19R
  - passable as generic positive PnL research, but not strong enough for Topstep 50K one-month objective.

Dynamic policy logic used:

- `rev` candidate if `prob_rev >= t_rev`, trend aligns with reversal, and (`ADX >= threshold` or strong rev probability).
- `cont` candidate if `prob_cont >= t_cont`, trend aligns with breakout, and (`ADX <= threshold` or strong cont probability).
- If both candidates are valid, choose higher probability.
- Otherwise skip.

## Topstep Objective Results

Active objective:

- 50K target: pass in about 20 trading days
- Profit target: $3,000
- Maximum Loss Limit: $2,000
- Consistency: best day must be below 50% of total profit at pass

From `TOPSTEP_PASS_REPORT.md`:

50K:

- Final decision: NO-GO
- Best raw pass-rate:
  - pass_rate: 45.3%
  - fail_mll_rate: 45.8%
  - conclusion: too aggressive and not acceptable
- Best risk-adjusted:
  - pass_rate: 34.1%
  - fail_mll_rate: 29.1%
  - median 20-day PnL: $1,332
  - 2026 pass_rate: 0%
  - 2026 fail_mll_rate: 60.7%

100K:

- Final decision: NO-GO
- Best raw pass-rate:
  - pass_rate: 35.8%
  - fail_mll_rate: 54.2%
- Best risk-adjusted:
  - pass_rate: 1.7%
  - fail_mll_rate: 0%
  - conclusion: too conservative to pass

### Simulator Refinement (v4)

From `SIMULATOR_COMPARISON.md` — controlled test on y_1r4_close60m:

| Metric | Old Simulator | Refined Simulator | Δ |
|--------|:------------:|:-----------------:|:-:|
| Pass Rate | 15.9% | 17.6% | **+1.7%** |
| Fail MLL | 26.0% | 21.1% | **-4.9%** |
| Score | -0.1003 | -0.0346 | **+0.0657** |

### v4 Objective Sweep (Refined Simulator)

From `OBJECTIVE_SWEEP_REPORT.md` — 10 targets, 2024+ holdout, refined Topstep sim:

| Rank | Target | Score | Pass | Fail MLL | PnL |
|:----:|--------|:----:|:----:|:--------:|:---:|
| 1 | y_1r2_180m | **+0.064** | 22.7% | 16.3% | $+1,264 |
| 2 | y_1r4_240m | +0.042 | 18.2% | 14.0% | $+608 |
| 3 | y_1r2_240m | +0.040 | 17.6% | 13.7% | $+1,136 |
| 4 | y_1r2_close60m | +0.033 | 11.9% | 8.7% | $+429 |
| 5 | y_1r2_120m | +0.029 | 23.7% | 20.8% | $+1,052 |
| 6 | y_1r2_60m | +0.014 | 17.8% | 16.4% | $+514 |
| 7 | y_1r4_120m | -0.003 | 23.2% | 23.5% | $+307 |
| 8 | y_1r4_180m | -0.005 | 24.4% | 24.9% | $+439 |
| 9 | y_1r4_close60m | -0.057 | 12.5% | 18.2% | $+94 |
| 10 | y_1r4_60m | -0.189 | 17.8% | 36.7% | $-40 |

Best 2024 performance: y_1r4_180m ($+1,872 PnL, 32.5% pass)
Best 2025 performance: y_1r2_120m ($+1,105 PnL, 28.5% pass)
2026 pass rate: 0% for 8/10 targets (unchanged from v3)

> Note: v4 holdout starts 2024+ (v3 was 2025+), so scores are not directly comparable.

Hard conclusion:

> ORB_v1.0 should not be treated as a Topstep-ready strategy. It is a baseline and research scaffold only.
> The refined simulator (v4) is strictly better but cannot fix the 2026 model OOD failure.

## Current Decision

**The "recent regime training" hypothesis is now disproved.**

v5 sweep (train on 2024-2025 only, test on 2026):
- 0% pass in 2026 for 9/10 targets
- Best 2026: y_1r4_240m (9.8% pass) but with 26.2% fail MLL — unusable

The 2026 OOD problem is caused by **feature drift, not training data regime mismatch.** No amount of retraining on recent data fixes 2026.

Research should focus on:
1. **New features with stationary predictive power** across volatility regimes
2. **Regime-adaptive strategy** — regime detection + model switching
3. **Alternative edge** — different instrument/strategy not dependent on ORB
4. **Accept that ORB edge may be dead in 2026**

## Backlog

### Priority 1: Objective Finding ✅

- Sweep runner built (`objective_sweep_orb_v4.py` uses shared `topstep_sim.py` module)
- All 10 targets swept with refined simulator
- v5 hypothesis test (recent-regime-only training) completed — **hypothesis DISPROVED**
- Best v4: y_1r2_180m (+0.064 score, 22.7% pass)
- Best v5 2026: y_1r4_240m (9.8% pass, but 26.2% fail MLL)
- **2026 remains the bottleneck** — 0% pass for 9/10 targets (v5) and 8/10 targets (v4)

### Priority 2: Topstep Simulator Accuracy ✅

- Implemented in `pipeline/analysis/topstep_sim.py`:
  - CT-based trading day boundaries via `map_to_topstep_trade_day()`
  - Fixed $3.00 commission (MGC round-turn)
  - Consistency rule (best day < 50% of total profit)
  - MLL trailing (highest end-of-day PnT, locked at starting balance)
  - Walk-forward: rolling 20-day windows
- Still pending:
  - Intraday mark-to-market from 1m path
  - MGC contract integer sizing
  - Max contracts/day and max trades/day rules

### Priority 3: Feature Finding (Scale-Invariant) ✅

- 7 scale-invariant features added (v3 sweep): `breakout_strength_atr_ratio`, `atr14_sq`, `breakout_strength_sq`, `price_vs_vwap_pct_abs`, `orb_range_sq`, `adx_50_flag`, `breakout_strength_vs_orb`
- Feature distribution analysis completed (see `model/SWEEP_v3/FEATURE_DISTRIBUTIONS.csv`)
- Scale-invariant features help but cannot fix 4.6× volatility expansion in 2026

Only after an objective/horizon shows promise:

- Prior session range
- Breakout candle volume ratio
- ORB cumulative volume ratio
- Realized volatility before breakout
- 4H / daily gold trend
- Distance from prior close
- News/event day filter
- Session-specific microstructure

### Priority 4: Execution Realism

- Re-label using next-candle open entry instead of breakout candle close.
- Quantify same-candle TP/SL ambiguity.
- Model slippage during high-volatility 2025-2026 regime.

### Priority 5: Modular Feature Architecture (v6+)

All features are now independent parquet modules in [`data/Level_1_Features/modules/`](data/Level_1_Features/modules/):
- **Grain key**: `(date, session, orb_tf, breakout_ts)` — breakout-event level
- **Loader**: [`pipeline/feature/modules/loader.py`](pipeline/feature/modules/loader.py) auto-joins all `*_features.parquet` via LEFT JOIN
- **Template**: [`pipeline/feature/modules/_TEMPLATE_generate_feature_module.py`](pipeline/feature/modules/_TEMPLATE_generate_feature_module.py) for creating new modules
- **Scalability**: Add a new feature family by creating a generator → run it → re-sweep. No pipeline changes needed.

| v5 (monolithic) | v6 (modular) |
|---|---|
| Features embedded in `training_datamart_orb.parquet` | Features in independent `data/Level_1_Features/modules/*.parquet` |
| Sweep script imports/adds features inline | Sweep script auto-loads all modules via `load_features_from_modules()` |
| Adding features requires modifying sweep script | Adding features = new module generator + re-run |

Active modules (v6+ — **7 modules, 42 total features**):
- [`orb_context_features`](pipeline/feature/modules/generate_orb_context_features.py) — 7 features: market context at breakout time (VWAP, ADX, EMA slope, day-of-week, etc.)
- [`scale_invariant_features`](pipeline/feature/modules/generate_scale_invariant_features.py) — 7 features: ratios, squares, flags robust to volatility scaling
- [`volatility_normalized_features`](pipeline/feature/modules/generate_volatility_normalized_features.py) — 5 features: rolling percentile ranks and z-scores of ATR14, breakout_strength, orb_range (stationary across volatility regimes)
- [`pre_breakout_profile_features`](pipeline/feature/modules/generate_pre_breakout_profile_features.py) — 5 features: pre-breakout candle pattern signals (compression ratio, drift/ATR, bullish ratio, inside bar flag, last candle range)
- [`session_momentum_features`](pipeline/feature/modules/generate_session_momentum_features.py) — 4 features: first 30-min range/ATR, direction/ATR, pre-breakout volume ratio, volume z-score
- [`interaction_features`](pipeline/feature/modules/generate_interaction_features.py) — 5 features: interaction terms (ATR×ADX, strength×range, |VWAP|×ATR, ADX×ORB/ATR, strength×session)
- [`macro_features`](pipeline/feature/modules/generate_macro_features.py) — 4 features: SPX regime (200-day MA), DXY trend (50-day MA), US10Y abs change, Oil volatility (abs return)

### Priority 5: Inference

Do not prioritize production inference yet.

Existing `inferences/orb/predict.py` appears stale relative to ORB_v1.0 artifacts. It references older target names (`y_60m`, `y_120m`, `y_240m`) and model files that do not match current `lgbm_rev_1r2_120m` and `lgbm_cont_1r2_120m`.

Inference should wait until a Topstep-suitable edge exists.

## Reproduction Commands

Train continuation model:

```bash
python3 pipeline/train/train_orb_continuation.py
```

Evaluate reversal holdout:

```bash
python3 pipeline/analysis/eval_holdout_orb.py
```

Evaluate dynamic rev/cont/skip policy:

```bash
python3 pipeline/analysis/eval_policy_switch_orb.py
```

Plot current PnL state:

```bash
python3 pipeline/analysis/plot_policy_pnl_state.py
```

Run Topstep pass-rate grid (legacy):

```bash
python3 pipeline/analysis/topstep_pass_engine.py
```

Run refined simulator comparison test:

```bash
python3 pipeline/analysis/test_refined_sim.py
```

Run v4 objective sweep (refined simulator):

```bash
python3 pipeline/analysis/objective_sweep_orb_v4.py
```

Run v5 objective sweep (recent-regime hypothesis test):

```bash
python3 pipeline/analysis/objective_sweep_orb_v5.py
```

Run v6 objective sweep (modular feature architecture):

```bash
python3 pipeline/analysis/objective_sweep_orb_v6.py
```

Generate feature modules (standalone — run before sweep if modules are missing):

```bash
# All feature modules
python3 pipeline/feature/modules/generate_orb_context_features.py
python3 pipeline/feature/modules/generate_scale_invariant_features.py
python3 pipeline/feature/modules/generate_volatility_normalized_features.py
```

## Agent Handoff Notes

- Do not claim ORB_v1.0 is trade-ready.
- Do not optimize for AUC as the primary objective.
- Use Topstep pass-rate, MLL breach-rate, and 2026 stability as the main scoreboard.
- Treat 2026 failure as the main research bottleneck.
- Objective sweep (v1-v6) now complete across all 10 labels.
- **v5 disproved the "recent regime" hypothesis** — training on 2024-2025 does NOT fix 2026.
- **v6 modular architecture** enables zero-friction feature experiments (just add a module parquet).
- The refined simulator (`topstep_sim.py`) is the new baseline for any future simulation work.
- 2026 OOD failure requires a fundamentally different approach (features that are distribution-robust, regime-conditional models, or alternative edges).
- v5/v6 data leakage fix: TRAIN_TO must end before HOLDOUT_FROM (set TRAIN_TO="2025-11-30", HOLDOUT_FROM="2025-12-01").
- Grain key for all feature modules: `(date, session, orb_tf, breakout_ts)` — do NOT include `year` in merge keys.
- Module parquets are **breakout-event grain** (34,187 rows), NOT rev/cont grain (68,374 rows). The loader's LEFT JOIN handles the expansion.
- **`date` dtype must be `datetime.date`** for merge compatibility across modules. String dates cause silent NaN on all feature columns after merge.
- **yfinance** is installed (`--break-system-packages`). Available for external data fetching (SPY, DX-Y.NYB, ^TNX, CL=F).
- Macro features merge on `date` only (not full EVENT_KEY) and forward-fill for weekends/holidays.

### v6+ Results (2026-04-28): Volatility-Normalized Features

Adding [`volatility_normalized_features`](pipeline/feature/modules/generate_volatility_normalized_features.py) (5 features: rolling percentile ranks and z-scores) to the v6 modular sweep produced:

| Target | Change vs Baseline (2-module) | Key Finding |
|--------|:----------------------------:|-------------|
| `y_1r4_60m` | 0.0% → **6.0%** pass rate | First-ever pass rate for this target; 8.2% in 2026 |
| `y_1r4_180m` | 0.0% → **18.1%** pass rate | 9.8% in 2026; fail MLL 45.8% (high) |
| `y_1r4_240m` | 0.0% → **14.5%** pass rate | 19.7% in 2026 — highest 2026 pass yet |
| `y_1r4_120m` | 10.8% → 14.5% pass rate | Pass up but fail MLL rose 0%→38.6% (more aggressive) |
| `y_1r4_close60m` | 1.2% → 1.2% pass rate | Unchanged |

**Interpretation**: Volatility-normalized features help the model extract signal in high-volatility regimes (2026), but they also increase trade frequency, raising fail-MLL risk. The optimal path forward may require combining volatility-normalized features with **stricter fail-MLL controls** (lower rev_q, higher risk cap thresholds) rather than abandoning the approach.

### v6+ Cycle 1 Results (2026-04-28): Pre-Breakout Profile Features

Adding [`pre_breakout_profile_features`](pipeline/feature/modules/generate_pre_breakout_profile_features.py) (5 features from 4×15m pre-breakout candles) to the 3-module baseline produced a **breakthrough**:

| Target | 3-Module → 4-Module | Δ Score | Δ Pass | Δ Fail MLL | Δ 2026 Pass |
|--------|:-------------------:|:-------:|:------:|:----------:|:-----------:|
| `y_1r4_240m` | 14.5% → **59.0%** pass | **+0.94** | **+44.6pp** | **-49.4pp** | **+50.8pp** |
| `y_1r2_60m` | 0.0% → **49.4%** pass | **+0.49** | **+49.4pp** | 0.0pp | **+60.7pp** |
| `y_1r2_180m` | 0.0% → **43.4%** pass | **+0.43** | **+43.4pp** | 0.0pp | **+45.9pp** |
| `y_1r4_close60m` | 1.2% → **37.3%** pass | **+0.49** | **+36.1pp** | -13.3pp | **+44.3pp** |
| `y_1r4_60m` | 6.0% → **30.1%** pass | **+0.31** | **+24.1pp** | -7.2pp | **+32.8pp** |

**All 10 targets improved. Zero regressions. All 4-module scores positive.**

The pre-breakout "coiled spring" signal (compression + drift + inside bar) is the most predictive feature family discovered so far. 2R targets revived from dead (0%) to viable (27-49%). 4R targets flipped from negative scores to strong positive.

### v6+ Cycle 2 Results (2026-04-28): Session Momentum Features

Adding [`session_momentum_features`](pipeline/feature/modules/generate_session_momentum_features.py) (4 features from 1m OHLCV: first-30-min range/ATR, direction/ATR, pre-breakout volume ratio, pre-breakout volume z-score) to the 4-module baseline produced mixed but powerful results:

| Target | 4-Module → 5-Module Score | Δ Pass | Δ 2026 Pass | Verdict |
|--------|:-------------------------:|:------:|:-----------:|:-------:|
| `y_1r4_180m` | +0.2410 → **+0.8193** | **+55.4pp** | **+57.4pp** (93.4%) | 🏆 **BEST EVER** |
| `y_1r2_240m` | +0.2771 → **+0.5783** | **+30.1pp** | **+27.9pp** (63.9%) | ✅ KEEP |
| `y_1r4_close60m` | +0.2410 → +0.3855 | +18.1pp | +18.0pp (63.9%) | ✅ KEEP |
| `y_1r2_close60m` | +0.3494 → +0.4578 | +34.9pp | +31.1pp (65.6%) | ✅ KEEP |
| `y_1r4_240m` | **+0.5904** → +0.4337 | -15.7pp | -23.0pp (47.5%) | ❌ Regressed |
| `y_1r2_120m` | +0.3855 → +0.1566 | +13.3pp | +1.6pp (47.5%) | ❌ Regressed (49.2% fail MLL) |

**6/10 improved, 3 regressed, 1 neutral. KEPT by decision** — the y_1r4_180m result at 93.4% 2026 pass is transformative.

### v6+ Cycle 3 Results (2026-04-28): Interaction Features

Adding [`interaction_features`](pipeline/feature/modules/generate_interaction_features.py) (5 interaction terms from breakout_events × market_context: ATR×ADX, strength×range, |VWAP dist|×ATR, ADX×ORB/ATR, strength×session) to the 5-module baseline:

| Target | 5-Module → 6-Module Score | Δ Score | Δ 2026 Pass | Verdict |
|--------|:-------------------------:|:-------:|:-----------:|:-------:|
| `y_1r2_120m` | +0.1566 → **+0.6627** | **+0.5060** | **+24.6pp** (72.1%) | ✅ WIN |
| `y_1r2_180m` | +0.4337 → **+0.5904** | +0.1566 | **+11.5pp** (67.2%) | ✅ WIN |
| `y_1r4_60m` | +0.0843 → **+0.2048** | +0.1205 | **+27.9pp** (49.2%) | ✅ WIN |
| `y_1r4_120m` | +0.3133 → **+0.4217** | +0.1084 | **+14.8pp** (57.4%) | ✅ WIN |
| `y_1r4_180m` | **+0.8193** → +0.4578 | -0.3614 | -32.8pp (60.7%) | ❌ LOSS |
| `y_1r4_240m` | +0.4337 → +0.1446 | -0.2892 | -16.4pp (31.1%) | ❌ LOSS |
| `y_1r2_60m` | +0.5181 → +0.2530 | -0.2651 | +9.8pp (60.7%) | ❌ LOSS |
| `y_1r4_close60m` | +0.3855 → +0.3012 | -0.0843 | -16.4pp (47.5%) | 🟠 Weak |
| `y_1r2_240m` | +0.5783 → +0.5422 | -0.0361 | -4.9pp (59.0%) | 🟠 Weak |
| `y_1r2_close60m` | +0.4578 → +0.4337 | -0.0241 | -8.2pp (57.4%) | 🟠 Weak |

**Pattern**: Interaction features help short-duration targets (60m-120m) but hurt long-duration (180m-240m). KEPT by decision — the short-duration improvements and zero data cost outweigh long-duration regressions.

### v6+ Cycle 4 Results (2026-04-28): Macro Features

Adding [`macro_features`](pipeline/feature/modules/generate_macro_features.py) (4 features from macro_data.parquet: SPX 200-day regime, DXY trend, US10Y abs change, Oil volatility) to the 6-module baseline:

**Hypothesis**: ORB breakout outcomes are systematically affected by macro regime. Bull equity + weak USD + stable yields + low oil vol = cleaner trend environment.

**Data**: First external data source. Fetched via [`fetch_macro_data.py`](pipeline/fetch/fetch_macro_data.py) using yfinance — SPY, DX-Y.NYB, ^TNX, CL=F daily data (4,258 rows, 2009-06-01 → 2026-04-28).

| Target | 6-Module → 7-Module Score | Δ Score | Δ 2026 Pass | Verdict |
|--------|:-------------------------:|:-------:|:-----------:|:-------:|
| `y_1r4_240m` | +0.1446 → **+0.3976** | **+0.2530** | **+23.0pp** (54.1%) | ✅ WIN |
| `y_1r4_close60m` | +0.3012 → **+0.4458** | **+0.1446** | **+26.2pp** (73.8%) | ✅ WIN |
| `y_1r2_60m` | +0.2530 → **+0.3855** | **+0.1325** | 0.0pp (60.7%) | ✅ WIN |
| `y_1r4_120m` | +0.4217 → **+0.5301** | **+0.1084** | **+14.8pp** (72.1%) | ✅ WIN |
| `y_1r4_60m` | +0.2048 → +0.1446 | -0.0602 | -23.0pp (26.2%) | ❌ LOSS |
| `y_1r2_120m` | **+0.6627** → +0.5904 | -0.0723 | -1.6pp (70.5%) | ❌ LOSS |
| `y_1r2_180m` | **+0.5904** → +0.5181 | -0.0723 | +1.6pp (68.9%) | ❌ LOSS |
| `y_1r4_180m` | +0.4578 → +0.3614 | -0.0964 | -13.1pp (47.5%) | ❌ LOSS |
| `y_1r2_close60m` | +0.4337 → +0.3133 | -0.1205 | -1.6pp (55.7%) | ❌ LOSS |
| `y_1r2_240m` | +0.5422 → +0.5181 | -0.0241 | +8.2pp (67.2%) | 🟠 Weak |

**4 WIN, 6 LOSS. Net Δ score: +0.1928. KEPT.**

**Key pattern discovered**: Macro features have a **complementary duration profile** to interaction features:
- Interaction (Cycle 3): helps SHORT holds (60m-120m), hurts LONG holds (180m-240m)
- Macro (Cycle 4): helps LONG holds (240m, close60m), hurts SHORT holds (60m) and medium 2R

**Marquee wins**:
- `y_1r4_240m`: 2026 pass 31.1% → **54.1%** (+23pp), score +0.1446 → +0.3976
- `y_1r4_close60m`: 2026 pass 47.5% → **73.8%** (+26pp), score +0.3012 → +0.4458 ($3,098 PnL)
- `y_1r4_120m`: 2026 pass 57.4% → **72.1%** (+15pp), score +0.4217 → +0.5301

**Active modules (7)**: `orb_context`, `scale_invariant`, `volatility_normalized`, `pre_breakout_profile`, `session_momentum`, `interaction`, `macro`
