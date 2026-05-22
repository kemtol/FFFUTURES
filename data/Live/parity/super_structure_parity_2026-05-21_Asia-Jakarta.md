# Super Structure Parity 2026-05-21 (Asia/Jakarta)

Window UTC: `2026-05-20T17:00:00+00:00` -> `2026-05-21T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | mode | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | CONS | 2026-05-21 00:10 | 4545.80 | 2026-05-21 00:10 | 4546.70 | 0.90 |  |  |
| PASS | Long | AGGR | 2026-05-21 02:40 | 4541.70 | 2026-05-21 02:40 | 4541.60 | -0.10 |  |  |
| PASS | Short | CONS | 2026-05-21 07:15 | 4522.00 | 2026-05-21 07:15 | 4520.90 | -1.10 |  |  |
| PASS | Short | AGGR | 2026-05-21 14:00 | 4514.60 | 2026-05-21 14:00 | 4515.90 | 1.30 |  |  |

## Signal Entry vs UI Entry
| severity | side | mode | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | Long | CONS | 2026-05-21 00:10 | 4545.80 | MISSING |  |  |  |  | MISSING_UI_ENTRY | no UI theoretical entry within 5min |
| CRITICAL | Long | AGGR | 2026-05-21 02:40 | 4541.70 | MISSING |  |  |  |  | MISSING_UI_ENTRY | no UI theoretical entry within 5min |
| CRITICAL | Short | CONS | 2026-05-21 07:15 | 4522.00 | 2026-05-21 02:30 | 2026-05-21 08:15 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-21 02:30->2026-05-21 08:15 |
| CRITICAL | Short | AGGR | 2026-05-21 14:00 | 4514.60 | MISSING |  |  |  |  | MISSING_UI_ENTRY | no UI theoretical entry within 5min |
