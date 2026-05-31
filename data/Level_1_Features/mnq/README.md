# MNQ Level 1 Features

Index for MNQ Level 1 feature/context artifacts. Keep strategy-specific files in
their own subfolders so ORB, pullback, and future MNQ experiments do not share a
flat namespace.

## Strategy Folders

| Folder | Status | Main artifact | Notes |
| --- | --- | --- | --- |
| `ORB/` | active baseline | `context.parquet` | M1 New York ORB context |
| `m1_pullback_scalper/` | scaffold | pending | M1 pullback idea, no L1 artifact yet |

## Namespace Contract

```text
data/Level_1_Features/mnq/<strategy>/
```

Do not write MNQ strategy artifacts to Gold/MGC paths such as
`data/Level_1_Features/super_structure_ml/`.
