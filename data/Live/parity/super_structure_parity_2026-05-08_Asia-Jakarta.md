# Super Structure Parity 2026-05-08 (Asia/Jakarta)

Window UTC: `2026-05-07T17:00:00+00:00` -> `2026-05-08T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Short | 2026-05-07 17:20 | 4716.40 | 2026-05-07 17:20 | 4715.70 | -0.70 |  |  |
| CRITICAL | Short | 2026-05-07 21:00 | 4697.00 | 2026-05-07 21:00 |  |  | ENTRY_REJECTED | Trading is currently unavailable. The instrument is not in an active trading status. |
| PASS | Long | 2026-05-08 02:10 | 4737.70 | 2026-05-08 02:10 | 4737.50 | -0.20 |  |  |
| PASS | Short | 2026-05-08 07:30 | 4716.40 | 2026-05-08 07:30 | 4714.40 | -2.00 |  |  |
| PASS | Long | 2026-05-08 13:40 | 4748.30 | 2026-05-08 13:40 | 4746.80 | -1.50 |  |  |
| PASS | Short | 2026-05-08 15:15 | 4717.60 | 2026-05-08 15:15 | 4716.90 | -0.70 |  |  |

## Signal Entry vs UI Entry
| severity | side | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | Short | 2026-05-07 17:20 | 4716.40 | 2026-05-07 15:55 | 2026-05-08 01:10 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-07 15:55->2026-05-08 01:10 |
| CRITICAL | Short | 2026-05-07 21:00 | 4697.00 | 2026-05-07 15:55 | 2026-05-08 01:10 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-07 15:55->2026-05-08 01:10 |
| PASS | Long | 2026-05-08 02:10 | 4737.70 | 2026-05-08 02:10 | 2026-05-08 07:25 | CLOSED | 0.00 | 0.00 |  | UI entry matched |
| PASS | Short | 2026-05-08 07:30 | 4716.40 | 2026-05-08 07:30 | 2026-05-08 12:40 | CLOSED | 0.00 | 0.00 |  | UI entry matched |
| PASS | Long | 2026-05-08 13:40 | 4748.30 | 2026-05-08 13:40 | 2026-05-08 15:15 | CLOSED | 0.00 | 0.00 |  | UI entry matched |
| PASS | Short | 2026-05-08 15:15 | 4717.60 | 2026-05-08 15:15 |  | OPEN | 0.00 | 0.00 |  | UI entry matched |
