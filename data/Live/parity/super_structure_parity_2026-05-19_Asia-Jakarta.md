# Super Structure Parity 2026-05-19 (Asia/Jakarta)

Window UTC: `2026-05-18T17:00:00+00:00` -> `2026-05-19T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | mode | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | CONS | 2026-05-18 19:10 | 4561.30 | 2026-05-18 19:10 | 4560.40 | -0.90 |  |  |
| CRITICAL | Long | CONS | 2026-05-18 20:20 | 4569.00 | 2026-05-18 20:20 |  |  | ENTRY_REJECTED | Trading is currently unavailable. The instrument is not in an active trading status. |
| PASS | Long | CONS | 2026-05-18 22:05 | 4574.70 | 2026-05-18 22:05 | 4575.30 | 0.60 |  |  |
| PASS | Short | CONS | 2026-05-19 01:05 | 4567.90 | 2026-05-19 01:05 | 4567.30 | -0.60 |  |  |
| PASS | Short | CONS | 2026-05-19 03:00 | 4550.30 | 2026-05-19 03:00 | 4550.60 | 0.30 |  |  |
| PASS | Short | CONS | 2026-05-19 13:35 | 4483.90 | 2026-05-19 13:35 | 4475.80 | -8.10 |  |  |

## Signal Entry vs UI Entry
| severity | side | mode | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | CONS | 2026-05-18 19:10 | 4561.30 | 2026-05-18 19:10 | 2026-05-19 00:30 | CLOSED | 0.00 | 0.00 |  | UI entry matched |
| CRITICAL | Long | CONS | 2026-05-18 20:20 | 4569.00 | 2026-05-18 19:10 | 2026-05-19 00:30 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-18 19:10->2026-05-19 00:30 |
| CRITICAL | Long | CONS | 2026-05-18 22:05 | 4574.70 | 2026-05-18 19:10 | 2026-05-19 00:30 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-18 19:10->2026-05-19 00:30 |
| PASS | Short | CONS | 2026-05-19 01:05 | 4567.90 | 2026-05-19 01:00 | 2026-05-19 08:00 | CLOSED | 5.00 | 1.70 |  | UI entry matched |
| CRITICAL | Short | CONS | 2026-05-19 03:00 | 4550.30 | 2026-05-19 01:00 | 2026-05-19 08:00 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-19 01:00->2026-05-19 08:00 |
| CRITICAL | Short | CONS | 2026-05-19 13:35 | 4483.90 | MISSING |  |  |  |  | MISSING_UI_ENTRY | no UI theoretical entry within 5min |
