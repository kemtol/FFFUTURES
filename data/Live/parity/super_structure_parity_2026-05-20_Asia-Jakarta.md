# Super Structure Parity 2026-05-20 (Asia/Jakarta)

Window UTC: `2026-05-19T17:00:00+00:00` -> `2026-05-20T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | mode | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Short | CONS | 2026-05-20 02:45 | 4462.90 | 2026-05-20 02:45 | 4463.60 | 0.70 |  |  |
| PASS | Long | CONS | 2026-05-20 11:30 | 4502.60 | 2026-05-20 11:30 | 4502.60 | 0.00 |  |  |

## Signal Entry vs UI Entry
| severity | side | mode | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | Short | CONS | 2026-05-20 02:45 | 4462.90 | 2026-05-20 01:40 | 2026-05-20 07:10 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-20 01:40->2026-05-20 07:10 |
| CRITICAL | Long | CONS | 2026-05-20 11:30 | 4502.60 | 2026-05-20 08:30 | 2026-05-20 13:40 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-20 08:30->2026-05-20 13:40 |
