# Super Structure Parity 2026-05-14 (Asia/Jakarta)

Window UTC: `2026-05-13T17:00:00+00:00` -> `2026-05-14T17:00:00+00:00`

Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.
Manual closes and theoretical exits are context, not critical parity failures.

## Signal Entry vs Topstep Entry
| severity | side | mode | signal_entry | signal_px | topstep_entry | topstep_px | slippage | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long |  | 2026-05-14 00:25 | 4708.70 | 2026-05-14 00:25 | 4707.80 | -0.90 |  |  |

## Signal Entry vs UI Entry
| severity | side | mode | signal_entry | signal_px | ui_entry | ui_exit | ui_status | entry_delta_min | entry_px_delta | drift_type | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | Long |  | 2026-05-14 00:25 | 4708.70 | 2026-05-14 00:20 | 2026-05-14 01:10 | CLOSED | 5.00 | 0.60 |  | UI entry matched |
