# Live Buffer Data Health — 2026-05-18 23:50 WIB

**Status: ⚠️ WARN**  (6/7 pass, 1 warn, 0 critical)

Buffer: `/home/kemal/futures/data/Live/topstepx_buffer.db`
Window: last `24h`

## Checks

| Check | Severity | Detail |
| --- | --- | --- |
| freshness | ✅ PASS | latest @ 2026-05-18T16:49:00+00:00 (101.9s ago) |
| quantity | ⚠️ WARN | 1118/1440 bars (77.6%) — only 77.6% of expected (market halt acceptable) |
| continuity | ✅ PASS | 2 gap(s), max 2.0min (effective 2.0min after CME halt) |
| ohlc_sanity | ✅ PASS | 0/1118 invalid |
| price_plausibility | ✅ PASS | close 4552.3 (range 1000.0-10000.0) |
| duplicate_timestamps | ✅ PASS | 0 dup row(s) |
| topstepx_account | ✅ PASS | account 22303383 CONNECTED |
