# Next Steps Plan — MGC ORB Edge Research

**Status:** ✅ **COMPLETED** — This was the original planning document from 2026-04-26. All phases have been executed. See [`plans/research_program.md`](plans/research_program.md) for the active research program, [`RETROSPECTIVE.md`](RETROSPECTIVE.md) for live inference calibration, and [`_MEMORY/20260502.md`](_MEMORY/20260502.md) for latest progress.

---

## What Was Accomplished

### Phase 1: Objective Sweep ✅
- v1-v6 objective sweeps completed across all 10 labels
- 2026 OOD identified as main bottleneck (v4: 0% pass for 8/10 targets)

### Phase 2: Topstep Simulator Refinement ✅
- CT-based trading day boundaries via `map_to_topstep_trade_day()`
- Fixed $3.00 commission (MGC round-turn)
- Consistency rule (best day < 50% of total profit)
- MLL trailing (highest end-of-day PnT, locked at starting balance)
- Walk-forward: rolling 20-day windows
- Refined simulator in [`pipeline/analysis/topstep_sim.py`](pipeline/analysis/topstep_sim.py)

### Phase 3: Feature Finding ✅
- v5 disproved "recent regime" hypothesis — feature drift, not training window
- v6 modular architecture built — zero-friction feature experiments
- **4 feature cycles completed** (all KEPT):
  - Cycle 1: Pre-Breakout Profile (5 features) — "coiled spring" signal, all 10 targets improved
  - Cycle 2: Session Momentum (4 features) — y_1r4_180m hit 93.4% 2026 pass
  - Cycle 3: Interaction Features (5 features) — helps short holds, hurts long holds
  - Cycle 4: Macro Features (4 features) — first external data (SPX, DXY, US10Y, Oil via yfinance), helps long holds
- **7 modules active, 42 features, all 10 targets positive**

### Phase 4: Execution Realism 🔄
- Not yet prioritized — feature discovery phase still ongoing

### Track B: Super Structure ML (V8 Router) ✅ LIVE since 2026-05-14
- **Status:** V8 router deployed to live `super_structure.service` (flag `USE_V8_ROUTER=True`).
- **Architecture:** Hybrid CONS ML (Meta-v7 Refined, dynamic per-session_cluster threshold) + AGGR mechanical (v1.12 pullback, `risk_pts <= 12` filter, no ML brain).
- **Walk-forward (sim) 90d:** PnL **+$5,151**, max DD **-$1,861** → PASS Topstep (borderline; $139 MLL headroom).
- **Sync verification:** `pipeline/super_structure_ml/eval/verify_router_sync.py` confirms router = sim trade-by-trade (0/0 divergence at 7d/30d/90d).
- **Monitoring stack realigned:** heartbeat shows V8 block (daily PnL vs $700 cap, mode, last decision); UI builder emits AGGR overlay; parity matches by `side + mode`.
- **Detail roadmap:** [`plans/super_structure_ml_v8.md`](plans/super_structure_ml_v8.md). Daily memory: [`_MEMORY/20260514.md`](_MEMORY/20260514.md).
- **Rollback:** flip `USE_V8_ROUTER=False` in `pipeline/live/super_structure.py:60` + restart service. Legacy SMART_1 dual-ML stack still loaded for instant fallback.
- **Next:** Monte Carlo combined sim (tail risk @ 90d), precision sharpening (`wick_ratio`/`atr_expansion` for sub-$1500 DD), feature-parity verification (live `smart_features.py` vs training builder).

---

## Current State (2026-04-28)

| Metric | Value |
|--------|-------|
| Active modules | 7 (`orb_context`, `scale_invariant`, `volatility_normalized`, `pre_breakout_profile`, `session_momentum`, `interaction`, `macro`) |
| Total features | 42 |
| Targets positive | 10/10 |
| Best score | `y_1r2_120m` +0.5904 (70.5% 2026 pass) |
| Best 2026 pass | `y_1r4_close60m` 73.8% ($3,098 PnL) |
| Best 2026 fail MLL | 0% on 6/10 targets |

## Key Discovery

**Duration-dependent effects are systematic and complementary:**
- Interaction features (Cycle 3): help SHORT holds (60m-120m), hurt LONG holds (180m-240m)
- Macro features (Cycle 4): help LONG holds (240m, close60m), hurt SHORT holds (60m)
- Together they cover the full duration spectrum

---

## Referensi

- Active research program: [`plans/research_program.md`](plans/research_program.md)
- Latest results: [`_MEMORY/20260428.md`](_MEMORY/20260428.md)
- PRD: [`_DOC/_PRD/0001_orb_edge.md`](_DOC/_PRD/0001_orb_edge.md)
- Modular architecture plan: [`plans/modular_feature_architecture.md`](plans/modular_feature_architecture.md)
- v6 sweep report: [`model/SWEEP_v6/OBJECTIVE_SWEEP_REPORT.md`](model/SWEEP_v6/OBJECTIVE_SWEEP_REPORT.md)
- v6 sweep results: [`model/SWEEP_v6/OBJECTIVE_SWEEP_RESULTS.csv`](model/SWEEP_v6/OBJECTIVE_SWEEP_RESULTS.csv)
