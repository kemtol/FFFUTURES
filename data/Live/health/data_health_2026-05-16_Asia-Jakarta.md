# Live Buffer Data Health — 2026-05-16 23:48 WIB

**Status: ✅ PASS**  (6/6 pass, 0 warn, 0 critical)
🏖️ **Market: CME CLOSED** — weekend halt (Sat)

Buffer: `/home/kemal/futures/data/Live/topstepx_buffer.db`
Window: last `24h`

## Checks

| Check | Severity | Detail |
| --- | --- | --- |
| freshness | ✅ PASS | latest @ 2026-05-15T20:58:00+00:00 (71411.9s ago) — stale 71412s but weekend halt (Sat) — expected |
| quantity | ✅ PASS | 250/1440 bars (17.4%) — 17.4% of nominal — weekend halt (Sat), expected |
| continuity | ✅ PASS | no gaps |
| ohlc_sanity | ✅ PASS | 0/250 invalid |
| price_plausibility | ✅ PASS | close 4544.1 (range 1000.0-10000.0) |
| duplicate_timestamps | ✅ PASS | 0 dup row(s) |
