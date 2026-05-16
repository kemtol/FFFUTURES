# Super Structure Parity 2026-05-12 (Asia/Jakarta)

Window UTC: `2026-05-11T17:00:00+00:00` -> `2026-05-12T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | 2026-05-12 00:30 | 4772.10 | 2026-05-12 00:30 | 4773.40 | 1.30 |  |  |
| PASS | Short | 2026-05-12 01:40 | 4734.30 | 2026-05-12 01:40 | 4735.90 | 1.60 |  |  |
| PASS | Short | 2026-05-12 03:35 | 4735.00 | 2026-05-12 03:35 | 4735.00 | 0.00 |  |  |
| PASS | Short | 2026-05-12 15:10 | 4680.50 | 2026-05-12 15:10 | 4679.00 | -1.50 |  |  |

## Signal Entry vs UI Entry
| severity | side | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | Long | 2026-05-12 00:30 | 4772.10 | 2026-05-11 17:35 | 2026-05-12 01:00 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-11 17:35->2026-05-12 01:00 |
| CRITICAL | Short | 2026-05-12 01:40 | 4734.30 | 2026-05-12 01:05 |  | OPEN |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-12 01:05->open |
| CRITICAL | Short | 2026-05-12 03:35 | 4735.00 | 2026-05-12 01:05 |  | OPEN |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-12 01:05->open |
| CRITICAL | Short | 2026-05-12 15:10 | 4680.50 | 2026-05-12 01:05 |  | OPEN |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-12 01:05->open |
