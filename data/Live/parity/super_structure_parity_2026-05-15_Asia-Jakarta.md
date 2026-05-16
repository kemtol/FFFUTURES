# Super Structure Parity 2026-05-15 (Asia/Jakarta)

Window UTC: `2026-05-14T17:00:00+00:00` -> `2026-05-15T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | mode | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Short | CONS | 2026-05-14 18:55 | 4674.00 | 2026-05-14 18:55 | 4674.40 | 0.40 |  |  |
| PASS | Short | CONS | 2026-05-15 05:20 | 4587.20 | 2026-05-15 05:20 | 4586.00 | -1.20 |  |  |
| PASS | Long | AGGR | 2026-05-15 13:15 | 4550.80 | 2026-05-15 13:15 | 4547.40 | -3.40 |  |  |

## Signal Entry vs UI Entry
| severity | side | mode | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | Short | CONS | 2026-05-14 18:55 | 4674.00 | 2026-05-14 13:45 | 2026-05-14 22:35 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-14 13:45->2026-05-14 22:35 |
| CRITICAL | Short | CONS | 2026-05-15 05:20 | 4587.20 | 2026-05-15 00:30 | 2026-05-15 10:55 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-15 00:30->2026-05-15 10:55 |
| PASS | Long | AGGR | 2026-05-15 13:15 | 4550.80 | 2026-05-15 13:15 | 2026-05-15 13:30 | CLOSED | 0.00 | 0.00 |  | UI entry matched |
