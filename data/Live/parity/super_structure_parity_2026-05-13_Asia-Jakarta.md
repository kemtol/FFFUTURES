# Super Structure Parity 2026-05-13 (Asia/Jakarta)

Window UTC: `2026-05-12T17:00:00+00:00` -> `2026-05-13T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | 2026-05-12 18:50 | 4699.30 | 2026-05-12 18:50 | 4699.20 | -0.10 |  |  |
| PASS | Short | 2026-05-13 02:50 | 4704.60 | 2026-05-13 02:50 | 4705.10 | 0.50 |  |  |
| PASS | Short | 2026-05-13 13:40 | 4681.20 | 2026-05-13 13:40 | 4683.10 | 1.90 |  |  |

## Signal Entry vs UI Entry
| severity | side | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | Long | 2026-05-12 18:50 | 4699.30 | 2026-05-12 18:35 | 2026-05-13 01:30 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-12 18:35->2026-05-13 01:30 |
| CRITICAL | Short | 2026-05-13 02:50 | 4704.60 | 2026-05-13 02:00 | 2026-05-13 06:10 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-13 02:00->2026-05-13 06:10 |
| CRITICAL | Short | 2026-05-13 13:40 | 4681.20 | 2026-05-13 12:35 | 2026-05-13 16:15 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-13 12:35->2026-05-13 16:15 |
