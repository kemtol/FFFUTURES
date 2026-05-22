# Live Buffer Data Health — 2026-05-17 03:56 WIB

**Status: ✅ PASS**  (6/6 pass, 0 warn, 0 critical)
🏖️ **Market: CME CLOSED** — weekend halt (Sat)

Buffer: `/home/kemal/futures/data/Live/topstepx_buffer.db`
Window: last `24h`

## Checks

| Check | Severity | Detail |
| --- | --- | --- |
| freshness | ✅ PASS | latest @ 2026-05-15T20:58:00+00:00 (86292.1s ago) — stale 86292s but weekend halt (Sat) — expected |
| quantity | ✅ PASS | 2/1440 bars (0.1%) — 0.1% of nominal — weekend halt (Sat), expected |
| continuity | ✅ PASS | no gaps |
| ohlc_sanity | ✅ PASS | 0/2 invalid |
| price_plausibility | ✅ PASS | close 4544.1 (range 1000.0-10000.0) |
| duplicate_timestamps | ✅ PASS | 0 dup row(s) |
