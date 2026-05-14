# Plan: Live Buffer Data Health Check (Phase 1)

## Context

V8 router (live since 2026-05-14, see [`super_structure_ml_v8.md`](super_structure_ml_v8.md))
membaca dari `data/Live/topstepx_buffer.db` (`ohlcv_1m` table) tiap 30s untuk
generate signal. Kalau data ini stale / gap / corrupt, live behavior langsung
drift dari validated sim — silent killer.

Existing monitoring:
- Heartbeat (5 min) — bar state + indicator + V8 router status.
- Parity check (15 min) — signal ↔ Topstep execution ↔ UI consistency.

Yang HILANG: continuity & quality validation di L0 Live buffer itu sendiri.
Heartbeat assume data is OK; parity assume signal generation is OK. Tidak ada
yang verify raw bar stream integrity.

Phase 1 scope (per user decision): **Live buffer only**. L0 Raw + L2 datamarts
(sterile research artifacts) di-defer ke Phase 2 kalau dibutuhkan.

## Design Decisions

| | Choice | Why |
|---|---|---|
| D1 Scope | Live buffer only (`topstepx_buffer.db.ohlcv_1m`) | V8 live cuma consume ini. Layer lain tidak block live. |
| D2 Severity action | **Telegram alert only**, no auto-halt, no flatten | User accept risk — visibility > auto-intervention. User bisa manual `/halt` kalau report jelek. |
| D3 Cadence | Every 15 min, full push tiap run | Konsisten dengan parity timer. Noisy tapi visibility tinggi. |

## Files

**Create:**
- `pipeline/live/data_health_check.py` — main script (read-only)
- `pipeline/run/super_structure_data_health.service` — systemd oneshot
- `pipeline/run/super_structure_data_health.timer` — every 15 min
- `data/Live/health/data_health_{YYYY-MM-DD}_{tz}.{json,md}` — output reports

**Read-only reference:**
- `data/Live/topstepx_buffer.db` (ohlcv_1m table)
- `pipeline/live/parity_super_structure.py` — template untuk struktur report + Telegram push
- `pipeline/live/buffer.py` — `DataBuffer` class untuk akses tabel (atau direct sqlite3 untuk read-only audit)
- `data/Live/telegram.env` — bot token + chat ID (re-use existing TELEGRAM_ENV_PATH pattern)

## Checks (Live Buffer)

### Freshness
- `latest_bar_ts` vs `now()`. **CRITICAL** kalau stale > 10 min selama market hours (CME 23/5, weekend OK).

### Quantity (last 24h)
- Expected ~1440 1m bars (24h × 60) minus CME weekend halt + maintenance windows.
- **WARN** kalau row count < 90% expected (e.g. < 1296 bars di window 24h non-weekend).

### Continuity (gap detection)
- Compute `epoch_ms.diff()` antara consecutive bars di last 24h.
- Expected delta = 60000 ms (1 min).
- **WARN**: gap 5–30 min (kemungkinan market halt / brief disconnect)
- **CRITICAL**: gap > 30 min selama market hours (suspected feed outage)

### OHLC sanity
- Untuk last 24h bars, check per-row:
  - `high >= max(open, close)` ✓
  - `low <= min(open, close)` ✓
  - `volume >= 0` ✓
  - `close > 0` ✓ (catch zeroed-out data)
  - `high >= low` ✓ (basic invariant)
- **CRITICAL** kalau ada row violating any (count + sample timestamps in report).

### Price plausibility
- Latest close should be in plausible MGC range (e.g. 1000–10000).
- **WARN** kalau outside. Hardcoded sanity range, not statistical.

### Duplicate timestamps
- Check `count(*) > count(distinct timestamp_utc)` di last 24h.
- **CRITICAL** kalau ada dup (feed re-sent atau buffer write bug).

## Report Structure

JSON output:
```json
{
  "generated_at_utc": "2026-05-14T07:15:00+00:00",
  "date": "2026-05-14",
  "timezone": "Asia/Jakarta",
  "buffer_path": "/home/kemal/futures/data/Live/topstepx_buffer.db",
  "window_hours": 24,
  "summary": {
    "status": "PASS" | "WARN" | "CRITICAL",
    "checks_run": 6,
    "checks_passed": 5,
    "checks_warn": 1,
    "checks_critical": 0
  },
  "freshness": {
    "severity": "PASS",
    "latest_bar_ts": "2026-05-14T07:04:00+00:00",
    "age_seconds": 65,
    "threshold_seconds": 600,
    "note": ""
  },
  "quantity": {...},
  "continuity": {...},
  "ohlc_sanity": {...},
  "price_plausibility": {...},
  "duplicate_timestamps": {...}
}
```

Markdown summary untuk Telegram push (compact):
```
🩺 *Data Health* `2026-05-14 14:15 WIB`
Status: *PASS* (5/6 checks ok)

✅ Freshness: 65s ago
✅ Quantity (24h): 1432 bars (99.4% expected)
⚠️ Continuity: 1 gap 7min @ 02:00 UTC (likely maintenance)
✅ OHLC sanity: 1432/1432 valid
✅ Price plausibility: 4705.8 in range
✅ No duplicate timestamps
```

## Telegram Push Logic

Per D3 decision: **push every run**, full summary. Use existing
`_telegram_send()` helper pattern from `parity_super_structure.py:917`. Re-use
`TELEGRAM_ENV_PATH = data/Live/telegram.env` and primary chat ID.

CLI flag: `--telegram {auto|never|always}`. Default `always` di systemd
service. `never` untuk local debugging.

## Implementation Skeleton

```python
# pipeline/live/data_health_check.py
import argparse, json, sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
BUFFER_DB = ROOT / "data" / "Live" / "topstepx_buffer.db"
OUT_DIR = ROOT / "data" / "Live" / "health"
TELEGRAM_ENV_PATH = ROOT / "data" / "Live" / "telegram.env"

FRESHNESS_THRESHOLD_S = 600     # 10 min
GAP_WARN_MIN = 5
GAP_CRITICAL_MIN = 30
PRICE_MIN = 1000.0
PRICE_MAX = 10000.0
QUANTITY_WARN_RATIO = 0.90

def load_recent_bars(hours: int = 24) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{BUFFER_DB}?mode=ro", uri=True, timeout=5)
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(hours=hours)
    df = pd.read_sql(
        "SELECT timestamp_utc, open, high, low, close, volume, epoch_ms "
        "FROM ohlcv_1m WHERE symbol='MICRO_GOLD' AND timeframe='1m' "
        "AND timestamp_utc >= ? ORDER BY epoch_ms",
        con, params=[start.strftime("%Y-%m-%d %H:%M:%S")]
    )
    con.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df

def check_freshness(df, now): ...
def check_quantity(df, hours): ...
def check_continuity(df): ...
def check_ohlc_sanity(df): ...
def check_price_plausibility(df): ...
def check_duplicates(df): ...

def main():
    args = ...
    df = load_recent_bars(24)
    report = {
        "freshness": check_freshness(df, datetime.now(timezone.utc)),
        "quantity": check_quantity(df, 24),
        "continuity": check_continuity(df),
        "ohlc_sanity": check_ohlc_sanity(df),
        "price_plausibility": check_price_plausibility(df),
        "duplicate_timestamps": check_duplicates(df),
    }
    summary = aggregate_summary(report)
    write_json(report, summary)
    write_markdown(report, summary)
    if args.telegram != "never":
        push_telegram(report, summary)
```

## systemd Wiring

```ini
# pipeline/run/super_structure_data_health.service
[Unit]
Description=Super Structure Live Buffer Data Health Check
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/home/kemal/futures
ExecStart=/usr/bin/python3 -u /home/kemal/futures/pipeline/live/data_health_check.py --telegram always
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:/home/kemal/futures/data/Live/super_structure_data_health.log
StandardError=append:/home/kemal/futures/data/Live/super_structure_data_health.log
```

```ini
# pipeline/run/super_structure_data_health.timer
[Unit]
Description=Run Super Structure Data Health Check every 15 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=15min
Persistent=true

[Install]
WantedBy=timers.target
```

Install:
```bash
mkdir -p ~/.config/systemd/user
cp pipeline/run/super_structure_data_health.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now super_structure_data_health.timer
```

## Verification

1. **Syntax + dry run:**
   ```bash
   python3 -m py_compile pipeline/live/data_health_check.py
   python3 pipeline/live/data_health_check.py --telegram never
   ```
   Expect: JSON di `data/Live/health/data_health_*.json` + console summary.

2. **Synthetic CRITICAL test** — temporarily lower `FRESHNESS_THRESHOLD_S` to
   1s, run, expect CRITICAL status. Revert.

3. **Verify Telegram delivery:**
   ```bash
   python3 pipeline/live/data_health_check.py --telegram always
   ```
   Cek Telegram bot received the 🩺 message.

4. **Verify systemd timer fires:**
   ```bash
   systemctl --user start super_structure_data_health.service
   systemctl --user list-timers super_structure_data_health.timer
   tail -20 data/Live/super_structure_data_health.log
   ```

5. **Compare with manual SQL query** untuk validate freshness/quantity numbers
   match what live actually sees.

## Out of Scope (Phase 2 candidates)

- **L0 Raw integrity** (`MGC_1m.db`, etc.) — cek datamart-relevant DBs.
- **L2 SS datamart anchor row counts** — verify `v3_final_training.parquet` =
  3643 rows and `v1_12_training_datamart.parquet` = 1471 rows (sterile —
  jangan berubah).
- **Cross-layer aggregation check** — 5m bars == aggregated 1m bars within
  tolerance.
- **DVC cache integrity** — `dvc status` integration.
- **Feed-source diff** — bandingkan L0 Live vs Databento backfill untuk last
  24h, flag price discrepancy > N ticks.
- **Auto-halt integration** — kalau Phase 1 berjalan dan critical scenarios
  jelas, upgrade D2 ke "auto-halt on CRITICAL". Decision deferred sampai
  ada data dari Phase 1.

## Risk & Rollback

- Health check **read-only** — tidak modify buffer atau state. Worst case:
  Telegram noise / failed run.
- Rollback: `systemctl --user disable --now super_structure_data_health.timer`.
- Tidak interaksi dengan live trading path (D2 = no halt).

## Future Hook Points

Saat Phase 2 nanti add auto-halt: gunakan `super_structure_state.json`
`halt` flag yang sudah ada — health check writer set `halt: true` +
`halt_reason: "data_health: <details>"`, live loop akan deteksi di next
30s cycle dan stop new entries. Same mechanism as Topstep violation halt
(`super_structure.py:1373`).
