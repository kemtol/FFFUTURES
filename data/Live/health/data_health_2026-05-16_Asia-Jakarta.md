# Live Buffer Data Health — 2026-05-16 08:13 WIB

**Status: ✅ PASS**  (6/6 pass, 0 warn, 0 critical)

Buffer: `/home/kemal/futures/data/Live/topstepx_buffer.db`
Window: last `24h`

## Checks

| Check | Severity | Detail |
| --- | --- | --- |
| freshness | ✅ PASS | latest @ 2026-05-15T20:58:00+00:00 (15326.3s ago) — stale 15326s but weekend halt (Fri 16:00 CT → Sun 17:00 CT) — expected |
| quantity | ✅ PASS | 1184/1440 bars (82.2%) — 82.2% of nominal — weekend halt (Fri 16:00 CT → Sun 17:00 CT), expected |
| continuity | ✅ PASS | 1 gap(s), max 2.0min (effective 2.0min after CME halt) |
| ohlc_sanity | ✅ PASS | 0/1184 invalid |
| price_plausibility | ✅ PASS | close 4544.1 (range 1000.0-10000.0) |
| duplicate_timestamps | ✅ PASS | 0 dup row(s) |
