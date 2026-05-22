# Super Structure Parity 2026-05-22 (Asia/Jakarta)

Window UTC: `2026-05-21T17:00:00+00:00` -> `2026-05-22T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | mode | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | CONS | 2026-05-21 17:15 | 4527.70 | 2026-05-21 17:15 | 4529.30 | 1.60 |  |  |
| PASS | Short | CONS | 2026-05-21 23:55 | 4533.40 | 2026-05-21 23:55 | 4533.70 | 0.30 |  |  |

## Signal Entry vs UI Entry
| severity | side | mode | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long | CONS | 2026-05-21 17:15 | 4527.70 | 2026-05-21 17:15 | 2026-05-21 23:45 | CLOSED | 0.00 | 0.00 |  | UI entry matched |
| CRITICAL | Short | CONS | 2026-05-21 23:55 | 4533.40 | MISSING |  |  |  |  | MISSING_UI_ENTRY | no UI theoretical entry within 5min |
