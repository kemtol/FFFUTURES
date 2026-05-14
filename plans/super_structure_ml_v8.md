# MMMACHINE Futures: Strategic Roadmap (Meta-v8 → V8 Router Live)

## 🚀 Current Phase: V8 Router LIVE (2026-05-14)

V8 router (Meta-v7 Refined CONS + v1.12 AGGR mechanical) di-deploy ke live
super_structure.service pukul 10:07 WIB pada 2026-05-14, setelah walk-forward
validated PASS Topstep dan sync-verified 0/0 divergence vs simulator.

**Toggle flag:** `USE_V8_ROUTER = True` di `pipeline/live/super_structure.py`.
Legacy SMART_1 dual-ML (regime_dispatcher + cons + aggr brain) tetap di-load
untuk rollback cepat (flip flag + restart).

### ✅ Validated Milestone: Meta-v7 Refined (Conservative path)
- **Sim (90d):** PnL +$2,427 / max DD -$1,442 ✅ PASS Topstep
- **Monte Carlo:** 16.94% prob_of_ruin, median DD -$1,271
- **Control:** Dynamic per-session_cluster thresholds `{0:0.50, 1:0.50, 2:0.45}` dari `inference_config_refined.json`

### ✅ Validated Milestone: V8 Combined (CONS ML + AGGR Mechanical)
Hybrid yang sebenarnya di-deploy live. Aggressive ML brain (legacy) dibuang
karena 42.6% prob_of_ruin di walk-forward.

| Window | PnL | Max DD | Topstep |
| --- | ---: | ---: | --- |
| 7d | -$276 | -$276 | ✅ |
| 30d | +$1,811 | -$276 | ✅ |
| 90d | **+$5,151** | -$1,861 | ✅ (borderline) |

90d DD -$1,861 cuma $139 headroom dari MLL -$2,000. Tail risk perlu Monte
Carlo follow-up sebelum bilang fully safe.

Artifacts:
- `pipeline/super_structure_ml/eval/simulate_cons_ml_aggr_mech.py`
- `model/SUPER_STRUCTURE/simulation-compare/SIM_CONS_ML_AGGR_MECH_MGC_{7,30,90}d.json`
- Parity verifier: `pipeline/super_structure_ml/eval/verify_router_sync.py` (0/0 divergence)

### ✅ Completed: Live Integration (was Active Task #1)
- [x] Replace `_get_ml_prediction` di `super_structure.py` dengan `InferenceRouter.route_cons` (dynamic thresholds dari `inference_config_refined.json`).
- [x] Tambah pullback event detector (`pipeline/live/pullback_detector.py`) untuk AGGR signal generation.
- [x] Replace AGGR ML brain dengan mechanical `risk_pts <= 12` filter.
- [x] Mode-tagged position state (`_position_mode`, `_tp_price`, `_entry_bar_ts`) + persistence.
- [x] AGGR-specific exit logic: fixed SL=ST±1pt, TP=1R, 100-bar timeout. CONS keeps trend-flip + trailing SL.
- [x] $700 combined daily cap (CT trading day, Topstep-aligned). Note: NOT $300 — sim validated dengan $700, lihat decision D4.
- [x] Single-queue position rule (D1): AGGR di-skip kalau CONS in-position dan sebaliknya.
- [x] Sync verifier confirms router behavior == sim behavior trade-by-trade.

### ✅ Completed: Monitoring Alignment (2026-05-14)
- [x] Heartbeat (`super_structure_executor.heartbeat`) tampilkan V8 block: daily PnL vs cap, mode chip, last decision (PASS/SKIP + reason).
- [x] Signal storage tag `mode`, `tp`, `v8_router` fields supaya parity bisa filter.
- [x] UI builder (`build_super_structure_trade_events.py`) generate AGGR pullback overlay di 5m output (102 events di window 3.5 bulan).
- [x] Parity (`parity_super_structure.py`) match by `side + mode`. JSON + Markdown + Telegram table tampilkan kolom mode.

### 🛠️ Remaining Tasks
1. **Monte Carlo combined sim** — Phase B yang sempat di-skip. 5000 iter dengan portfolio sequencing untuk verify tail risk di window 90d yang DD-nya borderline -$1,861.
2. **Daily PnL persistence across restart** — saat ini in-memory di router (reset tiap restart). Acceptable, tapi kalau jadi masalah, persist via state JSON.
3. **Precision Sharpening (was Active Task #2)** — experiment `wick_ratio` + `atr_expansion` untuk reduce 90d DD < -$1,500. Belum mulai.
4. **Automated daily audit (was Active Task #3)** — cron-job `topstep_auditor.py` → Telegram. Belum mulai. Status: existing systemd timer untuk parity check sudah cover 15-min audit, mungkin daily audit cukup tinggal tambah summary push.

### 📅 Timeline
- **May 10:** Meta-v7 Refined locked + documented.
- **May 11-13:** Meta-v8 research scaffolding, GMM regime detector, standardized training pipeline.
- **May 14:** **V8 router live**. Walk-forward validated combined, sync-verified, deployed, monitoring stack realigned.
- **TBD:** Monte Carlo combined sim + precision sharpening + 2026 OOD continued validation.
