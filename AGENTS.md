# AGENTS.md — MGC Futures Research

Python 3.12 research repo. **Dual-track:** (A) ORB-based ML research → Topstep 50K evaluation; (B) TV-strategy live auto-trade with Telegram + TopstepX execution. No formal build/test/lint/CI tooling. All scripts run with `python3` from project root.

## Architecture

```
data/
├── Level_0_Raw/         # Immutable SQLite: MGC_1m.db, MGC_5m.db, MGC_15m.db
├── Level_1_Features/    # Parquet: breakout_events (34,187 rows), market_context, orb_ranges, macro_data
│   └── modules/         # AUTO-DISCOVERED feature parquets (*_features.parquet)
├── Level_2_Datamart/    # training_datamart_orb.parquet (68,374 rows = 2× breakout events, rev+cont)
└── Live/                # Live trading state
    ├── combined_buffer.db    # ⭐ SINGLE SOURCE: MGC_1m.db (2010–2026) + topstepx_buffer.db (live)
    ├── topstepx_buffer.db    # Live TopstepX WS data (ohlcv_1m)
    ├── topstepx_token.json   # JWT for TopstepX API (refreshed via Playwright)
    ├── topstepx_creds.json   # TopstepX email + password
    ├── telegram.env          # Telegram bot token + chat ID
    ├── super_structure_signals.json  # Super Structure trade history (persisted)
    ├── fvg_signals.json              # FVG Scalper trade history (persisted)
    ├── super_structure.log           # Super Structure daemon stdout/stderr
    ├── tradingview_trades.json       # TradingView platform trade export (calibration source)
    ├── super_structure.pine          # Pine Script source for TV indicator
    ├── topstepx_feed.log     # Feed daemon log
    └── fvg_live.log          # FVG daemon log
pipeline/
├── live/                # 🔴 LIVE INFERENCE & EXECUTION
│   ├── super_structure.py  # Super Structure: ST+DEMA+ADX+CCI, V8 router gate (CONS+AGGR), check(), run_live()
│   ├── inference_router.py # V8 router: CONS ML (Meta-v7 Refined) + AGGR mechanical (v1.12), single-queue, $700 cap
│   ├── pullback_detector.py # V8 AGGR pullback event detector (mirrors v1.12 datamart rules)
│   ├── fvg_scalper.py      # FVG Scalper: FVG+DEMA+ADX+CHOP, check(), run_live() (parked)
│   ├── run_super_structure_live.py  # Entry-point for systemd daemon (single SuperStructure instance)
│   ├── signal_bus.py       # Pub/sub: _format_super_structure(), _format_fvg_scalper(), Telegram send
│   ├── user_db.py          # SQLite: users/chats/subscriptions, WAL mode
│   ├── buffer.py           # BufferManager: SQLite read/write, fill_range(), detect_gaps(), repair()
│   ├── runner.py           # ORB main daemon (research only, no Telegram)
│   ├── feature_builder.py  # Replicates batch 42-feature pipeline from buffer
│   ├── orb_detector.py     # Session management, ORB range, breakout detection
│   ├── portfolio.py        # Paper trading tracker with TP/SL auto-close
│   ├── bot.py              # Telegram bot: /status, /last, /pnl, /portfolio, /features
│   ├── rolling.py          # RollingStats manager (skeleton)
│   ├── run_feed.py         # TopstepX WS → SQLite daemon (auto-reconnect + gap repair)
│   ├── walkforward_telegram.py  # --batch (direct loop 2.98s) + --incremental modes
│   ├── calibrate_super_structure.py  # Python vs TradingView trade comparison
│   ├── send_super_structure_signals.py  # Direct backtest → Telegram sender
│   ├── execute/
│   │   └── super_structure_executor.py  # Super Structure Trade Executor: market entry, flatten, heartbeat, SL tracking
│   └── sources/
│       └── topstepx.py     # TopstepX WebSocket feed (RealTimeBar, SubscribeBars) + token fetch
├── research/             # Strategy backtest builders (separate from live execution)
│   ├── build_super_structure_trade_events.py    # Super Structure UI JSON/parquet generator
│   └── build_fvg_trade_events.py   # FVG Scalper UI JSON/parquet generator (params override)
├── feature/modules/      # Feature generators (standalone CLI) + loader.py + _TEMPLATE
├── analysis/             # Sweep runners, topstep_sim.py (shared), evaluation
│   └── sweep_fvg.py      # FVG parameter grid search
├── train/                # LGBM trainers (reversal + continuation)
├── fetch/                # yfinance data ingestion
└── run/                  # Daemon management
    ├── super_structure.service  # systemd user service for Super Structure
    └── restart_daemons.sh     # Post-reboot daemon launcher (cron @reboot)
model/                   # Trained models + sweep reports
_MEMORY/                 # Daily research logs (agent continuity)
inferences/orb/          # STALE — do not use
edges/orb_breakout/      # LEGACY — different target names, no Topstep sim
_ARCH/                   # DEAD CODE — FastAPI backtest, old strategies
ui/                      # Lightweight Charts SPA (multi-strategy: ST + FVG)
service/                 # EMPTY
```

## Strategies

### Super Structure (`super_structure`)

A trend-following multi-filter strategy on MGC 5m, ported from a TradingView Pine script. Live auto-trades through TopstepX. Single contract, market entries, code-managed trailing SL. The "edge" is filter-stacking: every condition must agree before entry, every condition individually can exit.

#### Fundamental — what it is

Indicators (all on 5m TF, resampled from 1m):

| Indicator | Params | Role |
|-----------|--------|------|
| SuperTrend | factor=4.0, ATR=12 | Regime (`direction` ±1) + dynamic SL anchor |
| DEMA | length=200 | Long-term trend filter; cross is entry trigger |
| ADX | length=12, threshold=25 | Strength gate |
| CCI | length=12, source=hl2 | Momentum trigger (>100 long, <−100 short) |

Entry logic (`super_structure.py:387-390`):
```
LONG  = ADX>25 ∧ CCI>+100 ∧ (cross_up_DEMA ∨ close>DEMA) ∧ ST_dir == −1
SHORT = ADX>25 ∧ CCI<−100 ∧ (cross_dn_DEMA ∨ close<DEMA) ∧ ST_dir == +1
```

Exit logic (`super_structure.py:392-435`): SL hit (`low ≤ sl` long / `high ≥ sl` short, `reason="SL"`) OR SuperTrend flip (`reason="TREND_FLIP"`). SL value = current SuperTrend, re-assigned each bar — pure trailing, code-only. **No exchange stop order.** If the daemon dies mid-position, the position has zero protection until the daemon comes back.

Calibration: 6/6 core trades match TradingView (Apr 29-30 2026); ~$0.20 average price delta from yfinance vs broker OHLC source — irreducible, not a bug. Internal subscription/publish key is `super_structure`. UI URL value: `?strategy=super_structure`.

Cost model: $10/point on MGC × 1 contract, commission **$1.74/round-turn** (TopstepX real: $0.87/leg × 2). Do not use $3.00 — that's the ORB sim number.

#### Objective — what it's trying to do

Hit a **Topstep 50K evaluation pass** (+$3,000 in ~20 trading days, MLL $2,000, consistency rule). The Super Structure → FVG pairing is intentional: ST is the magnitude play (lower WR, higher avg trade), FVG when it goes live will be the frequency play (higher WR, lower avg trade). Don't tune Super Structure for win rate; tune for expectancy and drawdown survival.

#### Architecture

```
TopstepX WS → run_feed.py → combined_buffer.db (3.6M bars 1m, append-only)
                                        ↓ poll every 30s
                              SuperStructure.check(now=…)
                                ├─ guard: skip if <30s since last check
                                ├─ incremental fetch: cache df, query new rows only
                                ├─ resample 1m → 5m (label="right", closed="left")
                                ├─ dedup: skip indicator compute if _last_ts unchanged
                                ├─ compute ST / DEMA / ADX / CCI
                                ├─ exit checks (SL, then trend flip)
                                ├─ entry checks (long/short)
                                └─ update _heartbeat_state
                                            ↓
                                _store_signal(action, price, sl, …)
                                ├─ append to self._signals (in-memory)
                                ├─ write super_structure_signals.json
                                ├─ SignalBus.publish("super_structure", payload)
                                │     └→ Telegram chat 7980136995
                                └─ executor.on_signal(payload)
                                      └→ SuperStructureExecutor
                                          ├─ _enter() → POST /Order (market, type=2)
                                          ├─ update_sl() → in-memory only (no API)
                                          └─ _exit() → DELETE /Position/close/{acct}
```

It's a **pull-based** loop, not push from WebSocket. Strategies poll the buffer; the WS daemon's only job is writing to SQLite. Performance comes from the 3-layer optimization in `check()` (per memory `_MEMORY/20260503.md`):

| Layer | Mechanism | Effect |
|-------|-----------|--------|
| 30s guard | skip if `now − _last_checked_now < 30s` | 0.4µs no-op |
| Dedup `_last_ts` | skip indicator compute if 5m bar unchanged | saves ~642ms |
| Incremental fetch | cache df, only query new rows since `_cached_end` | saves ~365ms |

In a 24h live run with 1-minute polling: ~101s total CPU = 0.12% utilization.

State variables (all in-memory, all reset on restart — see Footguns):

| Var | Lives in | Persists? |
|-----|----------|-----------|
| `_pos`, `_entry_price`, `_sl_price` | `SuperStructure` instance | ❌ |
| `_last_ts`, `_cached_df`, `_cached_end` | `SuperStructure` instance | ❌ |
| `_heartbeat_state` | `SuperStructure` instance | ❌ |
| `active`, `pos_side`, `entry_price`, `sl_price`, `sl_order_id` | `SuperStructureExecutor` | ❌ |
| `_signals[]` (history) | `super_structure_signals.json` | ✅ |
| `_seen` dedup LRU | `SignalBus` instance | ❌ |
| Subscriptions | `users.db` table | ✅ |

File map:

| File | Role |
|------|------|
| `pipeline/live/super_structure.py` | Strategy class, indicator funcs, `check()`, `run_live()` (poll loop + Telegram cmd handler) |
| `pipeline/live/execute/super_structure_executor.py` | TopstepX REST: `_enter`, `_exit`, `_flatten_all`, `heartbeat` |
| `pipeline/live/signal_bus.py` | Pub/sub, formatter `_format_super_structure`, dedup LRU(500), Telegram send |
| `pipeline/live/user_db.py` | SQLite subscription DB |
| `pipeline/live/buffer.py` | DataBuffer query layer over `combined_buffer.db` |
| `pipeline/live/run_super_structure_live.py` | systemd entrypoint |
| `pipeline/run/super_structure.service` | systemd unit (Restart=always, RestartSec=10) |
| `pipeline/research/build_super_structure_trade_events.py` | Backtest → parquet + UI JSON |
| `pipeline/live/calibrate_super_structure.py` | Compare Python vs UI backtest trades |
| `pipeline/live/compare_super_structure_trades.py` | Compare Python vs TradingView export (`tradingview_trades.json`) |
| `pipeline/live/walkforward_super_structure.py` | Walkforward sim (1 weekend window) |
| `pipeline/live/walkforward_telegram.py` | Walkforward + optional Telegram publish (`--batch` direct loop, `--incremental` simulates live) |
| `pipeline/live/send_super_structure_signals.py` | Backtest → publish to Telegram (replay tool) |
| `data/Live/super_structure.pine` | Pine source for the TV indicator (visual mirror of strategy) |

#### Operations — how it runs

| Aspect | Spec |
|--------|------|
| Process model | One systemd `--user` daemon per strategy. Two strategies = two daemons. |
| Service | `super_structure.service` → `python3 -u run_super_structure_live.py` |
| Restart | `Restart=always`, `RestartSec=10` (don't lower — rapid loop on crash) |
| Polling cadence | `check()` every 30s in `run_live` loop (`super_structure.py:600`) |
| Logs | `data/Live/super_structure.log` (append, **no rotation**) + `journalctl --user -u super_structure` |
| Heartbeat | Every 5 min via `executor.heartbeat(state)` — needs `bot_enabled` ∧ `_executor` |
| Token | `data/Live/topstepx_token.json`, manual refresh via Playwright |
| Telegram cfg | `data/Live/telegram.env` (TOKEN + CHAT_ID) |
| Telegram cmds | `/strat`, `/strat on/off <name>`, `/ss`, `/ss_status` |
| Subscribers | 1 user (`6283890722797` → chat `7980136995`) in `users.db` |
| Manual ops | `systemctl --user {start,stop,restart,status} super_structure` |
| Live tail | `journalctl --user -u super_structure -f` |

Common reasons the daemon will misbehave: token expired, Telegram rate-limit (409), DB lock from concurrent feed writer, missing `combined_buffer.db`, missing `telegram.env`, two listener processes (`restart_daemons.sh` + systemd both starting it).

#### Footguns (read before touching)

These are real, currently present in the code or known to bite:

1. **Duplicate exit block.** `super_structure.py:420-435` re-runs the SL + trend-flip exit checks already done at lines 392-418. The second block is dead code (because `_pos` is already 0), but it's confusing and any logic edit must be made in *both* places or you get drift.

2. **No state persistence across restart.** `_pos`, `_entry_price`, `_sl_price`, executor's `active` — all reset to 0/empty on daemon restart. If the exchange position is open and the daemon restarts, the strategy thinks it's flat and will happily place a fresh entry. **This was observed during the rename on 2026-05-06: a duplicate BUY was placed because of restart.** Any restart while in-position is a double-trade risk.

3. **No exchange stop order.** SL exists only as `_sl_price` in Python memory and the trailing SuperTrend value. If the process dies, the exchange position has no protective stop. The 10s `RestartSec` is a partial mitigation but not a guarantee.

4. **Single instance only.** `run_super_structure_live.py` calls `check()` once then `run_live()`. Two daemons = two executors = two orders per signal. Never let `restart_daemons.sh` and `systemctl start` both run it.

5. **`label="right"` is mandatory** on the 5m resample. `label="left"` shifts every timestamp 5 minutes behind TradingView's chart and silently produces wrong entries — calibration looks broken when it isn't.

6. **30s `check()` guard is not optional.** `_last_checked_now` prevents double-execution after a fast restart inside the 30s window. Don't remove the guard "just to be sure".

7. **DEMA(200) needs ≥120 days of warmup.** Shorter and DEMA hasn't converged → false signals on first bars. The warmup window is hardcoded in `check()` at `now − 120 days`.

8. **Database lock errors** during simultaneous feed write + strategy read are visible in logs. They self-recover; ignore unless they spike.

9. **Telegram 409 Conflict** in logs means another process is calling `getUpdates` with the same bot token. Likely cause: `restart_daemons.sh` left an orphan process, or FVG listener is running concurrently and polling the same bot.

10. **Log file grows without rotation.** `super_structure.log` is append-only. Rotate manually or set up `logrotate` before this fills the disk.

11. **Token refresh is manual.** When `topstepx_token.json` expires, the executor's API calls 401-fail silently into the `except` and the strategy keeps publishing signals to Telegram with no execution. Watch for "MARKET ... failed" in logs.

12. **Internal name vs file name confusion is gone, but TradingView-platform refs still exist.** `tradingview_trades.json`, `super_structure.pine` are TV-platform artifacts and intentionally keep TV/tradingview prefixes.

### FVG Scalper (`fvg_scalper`)
- **Indicators:** FVG (Fair Value Gap) + DEMA(200) + ADX(12) + CHOP(12)
- **Entry:** Gap >= MIN_GAP_PTS AND DEMA trend AND ADX > 25 AND CHOP < 48 AND session gating
- **SL:** Swing high/low (SL_LOOKBACK=15 bars)
- **TP:** 1.0× risk (TP_RISK_RATIO=1.0)
- **Sessions:** Asia + London only
- **Exit:** DEMA exit on loss only (DEMA_EXIT_ONLY_LOSS=True)
- **Telegram:** `📡 *FVG Scalper — Signal*` / `📡 *FVG Scalper — Exit*`
- **Status:** Fully ported, calibrated (31/32 trades match TV), parked — not yet live

## Critical Gotchas (miss these and everything breaks)

### ORB/ML track
- **`date` dtype must be `datetime.date`**, not `str`. String dates cause silent NaN on ALL feature columns after merge. Every module must assert `df["date"] = pd.to_datetime(df["date"]).dt.date`.
- **Module grain**: `(date, session, orb_tf, breakout_ts)` — one row per breakout event (34,187 rows). Do NOT include `year` or `side` in merge keys.
- **No look-ahead**: features must use only data available BEFORE `breakout_ts`.
- **Alphabetical merge order**: `loader.py` merges modules sorted by filename. Adding a module whose name sorts earlier shifts merge order (should be harmless, but be aware).
- **Column name conflicts**: if two modules define the same feature column, pandas adds `_x`/`_y` suffixes silently. The loader warns. Use `--force` when regenerating a module with intent.
- **Fixed commission**: $3.00/round-turn for ORB sim. MGC = $1.00/tick/contract.
- **Data leakage**: `TRAIN_TO` must end before `HOLDOUT_FROM`. Set `TRAIN_TO="2025-11-30"`, `HOLDOUT_FROM="2025-12-01"`.
- **Macro features merge on `date` only** (not full EVENT_KEY), forward-fill for weekends.

### Live trading track
- **`label="right"` in resample**: MUST use `label="right", closed="left"` for 5m bars. `label="left"` shifts timestamps 5 min behind TradingView — causes false entry signals and calibration mismatch.
- **Single SuperStructure instance**: `run_super_structure_live.py` must create ONE `SuperStructure()`, call `.check()` ONCE then `.run_live()`. Two instances = double-trade.
- **No exchange stop orders for Super Structure**: SL is code-only SuperTrend trail. Exit via `_flatten_all()` not stop order. Never send `POST /Order` with stop/target fields for this strategy.
- **Commission**: Super Structure live auto-trade uses $1.74/round-turn (TopstepX real: $0.87/leg × 2). ORB sim uses $3.00.
- **Warmup**: use 120 days for DEMA(200) convergence. Shorter warmup → DEMA not converged → false signals on first bars.
- **systemd over nohup**: Always use systemd `--user` service for daemons. `nohup`/`disown` die on terminal disconnect in this environment.
- **Combined buffer**: use `combined_buffer.db` for backtests (3.6M bars). Use `topstepx_buffer.db` for live feed.
- **`check()` 30s guard**: `_last_checked_now` prevents double-execution on restart. Don't remove.
- **`_last_ts` dedup**: Skip indicator compute if 5m bar unchanged. Saves 642ms per skip.
- **Topstep trading day**: US CT-based. `map_to_topstep_trade_day()` subtracts 15h10m to map UTC timestamps to 5PM CT → 3:10PM CT next day boundary.
- **ORB v2.0 is research-only**: No Telegram, no live execution, no subscribers in user_db.

## Dependencies

**In requirements.txt** (8 deps): `aiohttp`, `pandas`, `pyarrow`, `playwright`, `websockets`, `yfinance`, `databento`, `zstandard`

**Implicit** (used in code, NOT in requirements.txt): `lightgbm`, `numpy`, `scikit-learn` (edges/ only)

**yfinance** is installed with `--break-system-packages`. Available tickers: SPY, DX-Y.NYB, ^TNX, CL=F.

## Primary Commands

### Live Trading (systemd services)

```bash
# Super Structure daemon
systemctl --user start super_structure
systemctl --user stop super_structure
systemctl --user restart super_structure
systemctl --user status super_structure
journalctl --user -u super_structure -f      # live logs
journalctl --user -u super_structure -n 50   # last 50 lines

# Post-reboot restart all daemons
bash pipeline/run/restart_daemons.sh

# Feed daemon (manual start, usually via restart_daemons.sh)
python3 pipeline/live/run_feed.py            # TopstepX WS → combined_buffer.db

# UI backtest
cd ui && python3 -m http.server 4173
# → http://127.0.0.1:4173?strategy=super_structure|fvg_scalper&session=Asia,London&days=90
```

### Calibration & Testing

```bash
# Compare Python signals vs TradingView export
python3 pipeline/live/calibrate_super_structure.py

# Batch backtest → Telegram (dry-run)
python3 pipeline/live/walkforward_telegram.py --batch

# Incremental backtest (simulates live check() loop)
python3 pipeline/live/walkforward_telegram.py --incremental
```

### Strategy Backtest Builders

```bash
# Super Structure events for UI
python3 pipeline/research/build_super_structure_trade_events.py

# FVG events for UI (default best params)
python3 pipeline/research/build_fvg_trade_events.py

# FVG events with custom params
python3 pipeline/research/build_fvg_trade_events.py --params '{"MIN_GAP_PTS": 2.0, "SL_LOOKBACK": 20}'
```

### FVG Parameter Sweep

```bash
python3 pipeline/analysis/sweep_fvg.py           # phase 1 (broad)
python3 pipeline/analysis/sweep_fvg.py --phase 2 # phase 2 (refined)
```

### Feature Module Workflow (ORB/ML)

```bash
# Create new module from template
cp pipeline/orb_ml/features/modules/_TEMPLATE_generate_feature_module.py \
   pipeline/orb_ml/features/modules/generate_{family}_features.py

# Dry-run (verifies rows, NaN%, conflicts before writing)
python3 pipeline/orb_ml/features/modules/generate_{family}_features.py --dry-run

# Generate parquet
python3 pipeline/orb_ml/features/modules/generate_{family}_features.py [--force]

# Run active sweep (v6 modular — auto-discovers all modules)
python3 pipeline/orb_ml/analysis/objective_sweep_orb_v6.py
```

### Regenerate ORB feature modules

```bash
python3 pipeline/orb_ml/features/modules/generate_orb_context_features.py
python3 pipeline/orb_ml/features/modules/generate_scale_invariant_features.py
python3 pipeline/orb_ml/features/modules/generate_volatility_normalized_features.py
python3 pipeline/orb_ml/features/modules/generate_pre_breakout_profile_features.py
python3 pipeline/orb_ml/features/modules/generate_session_momentum_features.py
python3 pipeline/orb_ml/features/modules/generate_interaction_features.py
python3 pipeline/orb_ml/features/modules/generate_macro_features.py
```

### Training & Evaluation (ORB/ML)

```bash
python3 pipeline/orb_ml/train/train_orb_reversal.py
python3 pipeline/orb_ml/train/train_orb_continuation.py
python3 pipeline/orb_ml/train/train_orb_walk_forward_v2.py
python3 pipeline/orb_ml/analysis/eval_holdout_orb.py
python3 pipeline/orb_ml/analysis/eval_policy_switch_orb.py
python3 pipeline/analysis/plot_policy_pnl_state.py
python3 pipeline/analysis/test_refined_sim.py
python3 pipeline/analysis/eval_topstep_pass_v2.py
```

### Data Fetching

```bash
bash pipeline/run/run_fetch_mgc.sh
python3 pipeline/fetch/fetch_macro_data.py
python3 pipeline/orb_ml/features/build_orb_ranges.py
python3 pipeline/orb_ml/features/build_breakout_events.py
python3 pipeline/orb_ml/features/build_market_context.py
python3 pipeline/orb_ml/features/build_labels.py
```

## Scoring Metric (ORB/ML)

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

## TopstepX API Reference

| Action | Method | Endpoint | Notes |
|--------|--------|----------|-------|
| Entry | `POST /Order` | type=2 (market), positionSize=±1 | No stop/target for ST strategy |
| Exit | `DELETE /Position/close/{acct}` | Flatten all positions | 22303383 |
| Cancel | `DELETE /Order/cancel/{acct}/symbol/F.US.MGC` | Cancel all orders | Batch cancel |
| Token | Playwright browser login | `topstepx_token.json` | Refresh every few days |

## What NOT to do

- Do not claim ORB_v1.0 is trade-ready.
- Do not use `inferences/orb/predict.py` — stale, references wrong target names and model paths.
- Do not use code in `edges/orb_breakout/` — legacy naming conventions, incompatible grain.
- Do not use code in `_ARCH/` — dead experiments (FastAPI, vectorbt, old strategies).
- Do not create/run tests — no test framework exists. `test_refined_sim.py` is an ad-hoc comparison, not a test suite.
- Do not modify existing ORB feature modules — always create new ones.
- Do not add `year` to EVENT_KEY merge keys.
- Do not send exchange stop/limit orders for Super Structure — SL is code-only SuperTrend trail.
- Do not use `label="left"` in resample — timestamps shift 5 min behind TradingView.
- Do not create two SuperStructure instances — double-trade risk.
- Do not use `nohup`/`disown` for daemons — use systemd `--user` service.
- Do not start FVG listener until tested on paper (parked).
