# Super Structure Parity 2026-05-18 (Asia/Jakarta)

Window UTC: `2026-05-17T17:00:00+00:00` -> `2026-05-18T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | mode | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | CONS | 2026-05-18 13:30 | 4575.60 | 2026-05-18 13:30 | 4572.90 | -2.70 |  |  |
| PASS | Short | CONS | 2026-05-18 16:35 | 4544.10 | 2026-05-18 16:35 | 4544.90 | 0.80 |  |  |

## Signal Entry vs UI Entry
| severity | side | mode | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | Long | CONS | 2026-05-18 13:30 | 4575.60 | 2026-05-18 11:40 | 2026-05-18 14:40 | CLOSED |  |  | UI_ALREADY_IN_POSITION | UI theoretical was already in same-side trade 2026-05-18 11:40->2026-05-18 14:40 |
| CRITICAL | Short | CONS | 2026-05-18 16:35 | 4544.10 | MISSING |  |  |  |  | MISSING_UI_ENTRY | no UI theoretical entry within 5min |
