# PRD — Multi-Strategy Signal Hub (Phase 1)

## Objective
Run multiple trading strategies in one daemon. Each strategy detects signals independently, publishes to a central bus, and the bus routes to Telegram per subscribed user.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        DAEMON                            │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  ORB v2.0    │  │  TV Strategy │  │  Strategy N  │   │
│  │  (on/off)    │  │  (on)        │  │  (off)       │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │ signal          │ signal          │            │
│         ▼                 ▼                 ▼            │
│  ┌──────────────────────────────────────────────────┐    │
│  │                SIGNAL BUS                        │    │
│  │  subscribe(user_id, strategy_name)               │    │
│  │  publish(strategy_name, payload)                 │    │
│  └──────────────────┬───────────────────────────────┘    │
│                     │                                    │
│                     ▼                                    │
│  ┌──────────────────────────────────────────────────┐    │
│  │           NOTIFICATION LAYER                      │    │
│  │  (Telegram bot — one per chat_id)                 │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────┐  ┌───────────────────────────┐             │
│  │ Webhook  │  │ Telegram Command Polling   │             │
│  │ :8080    │  │ /status /last /pnl /port   │             │
│  └──────────┘  └───────────────────────────┘             │
└─────────────────────────────────────────────────────────┘

Users:
  6283890722797 → subscribe [tv_strategy, orb_v2]
  (future)      → subscribe [tv_strategy]
```

## Phase 1 Scope
**SignalBus → Telegram only.** No order execution yet. Phase 2 connects SignalBus → Order Management Service.

## Implementation Plan

### 1. New: `pipeline/live/signal_bus.py`
```
SignalBus (singleton)
├── subscribe(user_id, strategy_name)
├── publish(strategy_name, payload)
├── _format_orb_signal()      → Telegram markdown
├── _format_tv_signal()        → Telegram markdown
├── _send(user_id, text)       → raw Telegram API
└── _load_token()              → read data/Live/telegram.env
```

Seed subscription:
```python
HARDCODED_USER = "6283890722797"
{
  "tv_strategy": {HARDCODED_USER},
  "orb_v2":      {HARDCODED_USER},
}
```

### 2. Modify: `pipeline/live/tv_strategy.py`
- After `_store_signal()`, call `SignalBus().publish("tv_strategy", {...})`
- Payload: action, symbol, price, sl, reason, adx, cci

### 3. Modify: `pipeline/live/runner.py`
- In `_print_signal()`, replace `self.telegram.send(msg)` with `SignalBus().publish("orb_v2", signal)`
- Keep TelegramBot for command polling only

### 4. Modify: `pipeline/live/runner.py` entry point
- When `--live`: start `TVStrategy.run_live()` in a daemon thread alongside existing loop
- Single process, single Telegram bot, shared token

## Final Entry Point
```bash
python3 pipeline/live/runner.py --live
```

| Thread | Loop | Publish To |
|--------|------|------------|
| ORB v2.0 | 60s detect + predict | SignalBus → Telegram |
| TV Strategy | 30s check | SignalBus → Telegram |
| Webhook :8080 | HTTP server | → `_maybe_execute` → exec |
| Telegram command | 5s polling | /status /last /pnl /portfolio /features |

## Key Decisions
- One daemon, multiple strategy threads
- Hardcoded `chat_id = "6283890722797"` for Phase 1
- Telegram bot token from `data/Live/telegram.env`
- Subscriptions are in-memory dict (no persistence needed for Phase 1)
- Strategy on/off state per-strategy class (not in bus)
- `mark_as_written` trigger preserved (webhook still saves to `tv_signals.json`)

## Phase 2 (Out of Scope)
- SignalBus → Order Management Service
- Multi-user subscription management
- Strategy toggle via Telegram commands
- Subscription persistence
